// Workflow 04-matmul owns this file.
//
// ops::matmul is checked two ways: against ref::matmul on random inputs (cheap,
// any shape, including the tails a blocked kernel special-cases) and against
// the torch golden data (ground truth).

#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/allclose.h"
#include "common/oracle.h"
#include "common/random.h"
#include "common/thread_invariant.h"
#include "minillm/loader.h"
#include "minillm/ops.h"

using namespace minillm;
using namespace minillm::test;

namespace {

std::string shape_label(std::int64_t m, std::int64_t k, std::int64_t n) {
  return "m=" + std::to_string(m) + " k=" + std::to_string(k) + " n=" + std::to_string(n);
}

Tensor nan_filled(const Shape& shape) {
  Tensor t = Tensor::empty(shape);
  std::fill_n(t.f32(), t.numel(), std::numeric_limits<float>::quiet_NaN());
  return t;
}

/// An owning Q8_0 [n, k] weight built straight from blocks: every q uniform
/// in [-127, 127], every scale an f16 of random sign and mantissa with a
/// biased exponent in [7, 9], so |d| is in [2^-8, 2^-5) and the weights
/// q * d have about the spread of the N(0, 1) weights the F32 cases use.
Tensor random_q8_0(std::int64_t n, std::int64_t k, std::uint64_t seed) {
  Tensor t = Tensor::empty({n, k}, DType::Q8_0);
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> q(-127, 127);
  std::uniform_int_distribution<int> exponent(7, 9);
  std::uniform_int_distribution<int> mantissa(0, 1023);
  std::uniform_int_distribution<int> sign(0, 1);
  BlockQ8_0* blocks = t.data_as<BlockQ8_0>();
  for (std::int64_t b = 0; b < n * k / kQ8_0Block; ++b) {
    blocks[b].d = static_cast<std::uint16_t>((sign(rng) << 15) | (exponent(rng) << 10) | mantissa(rng));
    for (std::int8_t& v : blocks[b].qs) v = static_cast<std::int8_t>(q(rng));
  }
  return t;
}

/// An owning BF16 tensor holding f32_to_bf16 of every element of the F32
/// tensor src, as convert_to_bf16 (loader.h) computes it: lossless when every
/// value of src is a bf16 value.
Tensor to_bf16(const Tensor& src) {
  Tensor t = Tensor::empty(src.shape(), DType::BF16);
  std::uint16_t* dst = t.data_as<std::uint16_t>();
  for (std::int64_t i = 0; i < src.numel(); ++i) dst[i] = f32_to_bf16(src.f32()[i]);
  return t;
}

/// An owning BF16 [n, k] weight built straight from bit fields: a random sign
/// and 7-bit mantissa, and a biased exponent in [121, 128], so |w| is in
/// [2^-6, 4) with a mean square of about 1.5, about the spread of the N(0, 1)
/// weights the F32 cases use. Four values per draw keep the 49152-row lm_head
/// cheap to build under the sanitizers.
Tensor random_bf16(std::int64_t n, std::int64_t k, std::uint64_t seed) {
  Tensor t = Tensor::empty({n, k}, DType::BF16);
  std::mt19937_64 rng(seed);
  std::uint16_t* p = t.data_as<std::uint16_t>();
  std::uint64_t bits = 0;
  for (std::int64_t i = 0; i < t.numel(); ++i) {
    if (i % 4 == 0) bits = rng();
    const auto r = static_cast<std::uint16_t>(bits >> (16 * (i % 4)));
    p[i] = static_cast<std::uint16_t>((r & 0x807fu) | ((121u + ((r >> 7) & 7u)) << 7));
  }
  return t;
}

/// ops::matmul(x, w) for a Q8_0 w, into a NaN-filled out so that an element
/// the kernel skips cannot pass, checked against ref::matmul(x,
/// convert_to_f32(w)). Returns ops::matmul's output.
Tensor check_q8_0_matmul(const Tensor& x, const Tensor& w, const std::string& label) {
  Tensor actual = nan_filled({x.dim(0), w.dim(0)});
  ops::matmul(x, w, actual);
  Tensor expected = Tensor::zeros({x.dim(0), w.dim(0)});
  ref::matmul(x, convert_to_f32(w), expected);
  assert_allclose(actual, expected, tol::kMatmul, label);
  return actual;
}

/// Like MINILLM_SKIP_WITHOUT_ORACLE, for the checkpoint. Returns true,
/// recording the skip, if it is missing.
bool skip_without_checkpoint(const std::filesystem::path& file) {
  std::error_code ec;
  if (std::filesystem::is_regular_file(file, ec)) return false;
  const std::string reason =
      "missing checkpoint: " + file.string() + " — run `python oracle/generate.py`";
  note_skip(reason);
  MESSAGE("SKIP: " << reason);
  return true;
}

/// The same for a BF16 w.
Tensor check_bf16_matmul(const Tensor& x, const Tensor& w, const std::string& label) {
  Tensor actual = nan_filled({x.dim(0), w.dim(0)});
  ops::matmul(x, w, actual);
  Tensor expected = Tensor::zeros({x.dim(0), w.dim(0)});
  ref::matmul(x, convert_to_f32(w), expected);
  assert_allclose(actual, expected, tol::kMatmul, label);
  return actual;
}

}  // namespace

TEST_CASE("ops::matmul agrees with ref::matmul at the model's shapes") {
  // [m, k] x [n, k] for every linear layer in the forward pass, at m = 1 (the
  // decode step) and m = 8 (a short prefill). o_proj has q_proj's shape and
  // gate_proj has up_proj's, so each distinct shape appears once. lm_head uses
  // the full 49152-row vocabulary; only the golden file trims it.
  struct Case {
    const char* name;
    std::int64_t k;
    std::int64_t n;
  };
  const Case cases[] = {
      {"q_proj/o_proj", 576, 576},     {"k_proj/v_proj", 576, 192},
      {"gate_proj/up_proj", 576, 1536}, {"down_proj", 1536, 576},
      {"lm_head", 576, 49152},
  };

  for (const std::int64_t m : {1, 8}) {
    for (const Case& c : cases) {
      const Tensor x = random_tensor({m, c.k}, 201);
      const Tensor w = random_tensor({c.n, c.k}, 202);
      Tensor expected = Tensor::zeros({m, c.n});
      Tensor actual = Tensor::zeros({m, c.n});

      ref::matmul(x, w, expected);
      ops::matmul(x, w, actual);
      assert_allclose(actual, expected, tol::kMatmul,
                      std::string("ops::matmul vs ref ") + c.name + " " +
                          shape_label(m, c.k, c.n));
    }
  }
}

TEST_CASE("ops::matmul is bitwise identical at every thread count") {
  // Threads split the output features into whole column panels, so every
  // element keeps its single-thread accumulation order: not close, equal.
  struct Case {
    const char* name;
    std::int64_t m;
    std::int64_t k;
    std::int64_t n;
  };
  const Case cases[] = {
      {"prefill q_proj/o_proj", 128, 576, 576},
      {"prefill k_proj/v_proj", 128, 576, 192},
      {"prefill gate_proj/up_proj", 128, 576, 1536},
      {"prefill down_proj", 128, 1536, 576},
      {"decode q_proj/o_proj", 1, 576, 576},
      {"decode k_proj/v_proj", 1, 576, 192},
      {"decode gate_proj/up_proj", 1, 576, 1536},
      {"decode down_proj", 1, 1536, 576},
      {"lm_head", 1, 576, 49152},
      {"n = 1", 5, 37, 1},
      {"n = 7", 128, 576, 7},
      {"n = 49152, a row tail", 3, 64, 49152},
      {"a leftover column in a threaded split", 7, 33, 1537},
  };
  for (const Case& c : cases) {
    const std::string label =
        std::string("ops::matmul ") + c.name + " " + shape_label(c.m, c.k, c.n);
    const Tensor x = random_tensor({c.m, c.k}, 211);
    const Tensor w = random_tensor({c.n, c.k}, 212);
    const Tensor single = check_thread_invariant(label, [&] {
      Tensor out = nan_filled({c.m, c.n});
      ops::matmul(x, w, out);
      return out;
    });
    Tensor expected = Tensor::zeros({c.m, c.n});
    ref::matmul(x, w, expected);
    assert_allclose(single, expected, tol::kMatmul, label + " vs ref");
  }
}

TEST_CASE("ops::matmul handles awkward sizes") {
  // Sizes that are not multiples of any plausible vector width or tile, which
  // is where a hand-written kernel's tail handling goes wrong.
  const std::int64_t sizes[] = {1, 2, 3, 5, 7, 9, 15, 17, 31, 33};
  for (const std::int64_t k : sizes) {
    for (const std::int64_t n : sizes) {
      const Tensor x = random_tensor({3, k}, 203);
      const Tensor w = random_tensor({n, k}, 204);
      Tensor expected = Tensor::zeros({3, n});
      Tensor actual = Tensor::zeros({3, n});
      ref::matmul(x, w, expected);
      ops::matmul(x, w, actual);
      assert_allclose(actual, expected, tol::kMatmul,
                      "ops::matmul vs ref k=" + std::to_string(k) + " n=" + std::to_string(n));
    }
  }
}

TEST_CASE("ops::matmul handles every row count") {
  // The sweep above fixes m = 3. A kernel that blocks rows has a row tail of
  // its own, so walk m past every plausible row block, at a k with both a
  // vector body and a remainder and an n that leaves a column tail as well.
  struct Inner {
    std::int64_t k;
    std::int64_t n;
  };
  const Inner inners[] = {{37, 19}, {576, 64}};
  std::vector<std::int64_t> rows;
  for (std::int64_t m = 1; m <= 17; ++m) rows.push_back(m);
  for (const std::int64_t m : {31, 33, 64}) rows.push_back(m);

  for (const Inner& in : inners) {
    for (const std::int64_t m : rows) {
      const Tensor x = random_tensor({m, in.k}, 205);
      const Tensor w = random_tensor({in.n, in.k}, 206);
      Tensor expected = Tensor::zeros({m, in.n});
      Tensor actual = Tensor::zeros({m, in.n});
      ref::matmul(x, w, expected);
      ops::matmul(x, w, actual);
      assert_allclose(actual, expected, tol::kMatmul,
                      "ops::matmul vs ref " + shape_label(m, in.k, in.n));
    }
  }
}

TEST_CASE("ops::matmul overwrites every output element") {
  // out is fully overwritten, never accumulated into. Starting from NaN means
  // an element that is skipped, or added to instead of stored, cannot pass.
  // k = 0 is an empty sum, which is 0.
  struct Case {
    std::int64_t m;
    std::int64_t k;
    std::int64_t n;
  };
  const Case cases[] = {{1, 576, 576}, {8, 1536, 576}, {5, 9, 7}, {3, 0, 4}};

  for (const Case& c : cases) {
    const Tensor x = random_tensor({c.m, c.k}, 207);
    const Tensor w = random_tensor({c.n, c.k}, 208);
    Tensor expected = Tensor::zeros({c.m, c.n});
    Tensor actual = Tensor::empty({c.m, c.n});
    std::fill_n(actual.f32(), actual.numel(), std::numeric_limits<float>::quiet_NaN());

    ref::matmul(x, w, expected);
    ops::matmul(x, w, actual);
    assert_allclose(actual, expected, tol::kMatmul,
                    "ops::matmul over NaN " + shape_label(c.m, c.k, c.n));
  }
}

TEST_CASE("ops::matmul rejects invalid arguments") {
  const Tensor x = Tensor::zeros({2, 3});
  Tensor out = Tensor::zeros({2, 4});

  // The cases ref::matmul is held to (tests/ops_ref/test_matmul.cpp).
  // w's second dimension is in_features and must match x's width.
  CHECK_THROWS_AS(ops::matmul(x, Tensor::zeros({4, 5}), out), std::invalid_argument);
  // out must be [m, n].
  CHECK_THROWS_AS(ops::matmul(x, Tensor::zeros({5, 3}), out), std::invalid_argument);
  // rank 3 is not a matrix.
  CHECK_THROWS_AS(ops::matmul(Tensor::zeros({2, 2, 2}), Tensor::zeros({2, 2}), out),
                  std::invalid_argument);

  // out has the wrong number of rows.
  Tensor out_rows = Tensor::zeros({3, 4});
  CHECK_THROWS_AS(ops::matmul(x, Tensor::zeros({4, 3}), out_rows), std::invalid_argument);

  // x and out must be f32. w may also be Q8_0 or BF16, the weight formats
  // (their cases are below), but no other dtype.
  Tensor out_bf16 = Tensor::zeros({2, 4}, DType::BF16);
  CHECK_THROWS_AS(ops::matmul(Tensor::zeros({2, 3}, DType::BF16), Tensor::zeros({4, 3}), out),
                  std::invalid_argument);
  CHECK_THROWS_AS(ops::matmul(x, Tensor::zeros({4, 3}, DType::F16), out), std::invalid_argument);
  CHECK_THROWS_AS(ops::matmul(x, Tensor::zeros({4, 3}), out_bf16), std::invalid_argument);

  // Every operand must be contiguous: a column slice keeps its parent's row
  // stride, so it is not.
  Tensor out_view = Tensor::zeros({2, 8}).slice(1, 0, 4);
  CHECK_THROWS_AS(ops::matmul(Tensor::zeros({2, 6}).slice(1, 0, 3), Tensor::zeros({4, 3}), out),
                  std::invalid_argument);
  CHECK_THROWS_AS(ops::matmul(x, Tensor::zeros({4, 6}).slice(1, 0, 3), out),
                  std::invalid_argument);
  CHECK_THROWS_AS(ops::matmul(x, Tensor::zeros({4, 3}), out_view), std::invalid_argument);
}

TEST_CASE("ops::matmul matches torch") {
  for (const std::string variant : {"small", "qkv", "kv", "mlp_up", "mlp_down", "lm_head"}) {
    const std::string base = "ops/matmul_" + variant + "_";
    MINILLM_SKIP_WITHOUT_ORACLE({base + "x.npy", base + "w.npy", base + "out.npy"});

    const Tensor x = oracle_npy(base + "x.npy").to_tensor();
    const Tensor w = oracle_npy(base + "w.npy").to_tensor();
    const Tensor expected = oracle_npy(base + "out.npy").to_tensor();

    Tensor actual = Tensor::zeros(expected.shape());
    ops::matmul(x, w, actual);
    assert_allclose(actual, expected, tol::kMatmul, "ops::matmul " + variant);
  }
}

// ---------------------------------------------------------------------------
// Q8_0 weights (workflow 16). A Q8_0 w stands for exactly convert_to_f32(w),
// so every case is held to ref::matmul(x, convert_to_f32(w)) with the F32
// kernel's tolerance. m < 6 runs the decode kernel, m >= 6 the panel kernel
// (src/ops/q8_0.cpp); the decode kernel takes 4 rows of W at a time (2 for
// m > 1) and the panel kernel 24, so n = 7 and n = 50 leave a tail for both.
// ---------------------------------------------------------------------------

TEST_CASE("ops::matmul with a Q8_0 weight agrees with ref::matmul on convert_to_f32(w)") {
  for (const std::int64_t m : {1, 2, 3, 4, 5, 8, 17, 64, 79}) {
    for (const std::int64_t k : {32, 64, 576, 1536}) {
      for (const std::int64_t n : {1, 7, 50}) {
        const Tensor x = random_tensor({m, k}, 231);
        const Tensor w = random_q8_0(n, k, 232);
        check_q8_0_matmul(x, w, "ops::matmul q8_0 vs ref " + shape_label(m, k, n));
      }
    }
  }
}

TEST_CASE("ops::matmul with a Q8_0 weight agrees with ref::matmul at the model's shapes") {
  // Every projection shape at m = 1 (decode) and m = 79 (the longest oracle
  // prompt), and the full 49152-row lm_head, which only ever sees one row.
  struct Case {
    const char* name;
    std::int64_t m;
    std::int64_t k;
    std::int64_t n;
  };
  const Case cases[] = {
      {"q_proj/o_proj", 1, 576, 576},       {"k_proj/v_proj", 1, 576, 192},
      {"gate_proj/up_proj", 1, 576, 1536},  {"down_proj", 1, 1536, 576},
      {"lm_head", 1, 576, 49152},           {"prefill q_proj/o_proj", 79, 576, 576},
      {"prefill k_proj/v_proj", 79, 576, 192}, {"prefill gate_proj/up_proj", 79, 576, 1536},
      {"prefill down_proj", 79, 1536, 576},
  };
  for (const Case& c : cases) {
    const Tensor x = random_tensor({c.m, c.k}, 233);
    const Tensor w = random_q8_0(c.n, c.k, 234);
    check_q8_0_matmul(x, w, std::string("ops::matmul q8_0 vs ref ") + c.name + " " +
                                shape_label(c.m, c.k, c.n));
  }
}

TEST_CASE("ops::matmul with a Q8_0 weight is bitwise identical at every thread count") {
  // Threads split W into whole tiles (decode) or panels (prefill), and every
  // element is computed the same way whatever the split.
  struct Case {
    const char* name;
    std::int64_t m;
    std::int64_t k;
    std::int64_t n;
  };
  const Case cases[] = {
      {"decode q_proj/o_proj", 1, 576, 576},
      {"decode k_proj/v_proj", 1, 576, 192},
      {"decode gate_proj/up_proj", 1, 576, 1536},
      {"decode down_proj", 1, 1536, 576},
      {"lm_head", 1, 576, 49152},
      {"decode kernel, several rows of x", 5, 576, 576},
      {"prefill gate_proj/up_proj", 64, 576, 1536},
      {"prefill down_proj", 79, 1536, 576},
      {"a tail panel in a threaded split", 17, 64, 1537},
      {"n = 1", 8, 32, 1},
  };
  for (const Case& c : cases) {
    const std::string label =
        std::string("ops::matmul q8_0 ") + c.name + " " + shape_label(c.m, c.k, c.n);
    const Tensor x = random_tensor({c.m, c.k}, 235);
    const Tensor w = random_q8_0(c.n, c.k, 236);
    const Tensor single = check_thread_invariant(label, [&] {
      Tensor out = nan_filled({c.m, c.n});
      ops::matmul(x, w, out);
      return out;
    });
    Tensor expected = Tensor::zeros({c.m, c.n});
    ref::matmul(x, convert_to_f32(w), expected);
    assert_allclose(single, expected, tol::kMatmul, label + " vs ref");
  }
}

TEST_CASE("ops::matmul with a Q8_0 weight handles empty dimensions") {
  // k = 0 is an empty sum: every element is 0, even over a NaN-filled out.
  Tensor out = nan_filled({3, 4});
  ops::matmul(Tensor::zeros({3, 0}), Tensor::zeros({4, 0}, DType::Q8_0), out);
  for (std::int64_t i = 0; i < out.numel(); ++i) CHECK(out.f32()[i] == 0.0f);

  // No rows of x, or no rows of W: nothing to write, nothing to read.
  Tensor no_rows = Tensor::zeros({0, 4});
  ops::matmul(Tensor::zeros({0, 64}), random_q8_0(4, 64, 237), no_rows);
  Tensor no_cols = Tensor::zeros({3, 0});
  ops::matmul(random_tensor({3, 64}, 238), Tensor::zeros({0, 64}, DType::Q8_0), no_cols);
}

TEST_CASE("ops::matmul with a Q8_0 weight rejects invalid arguments") {
  // The same rules as for an F32 w, with errors that name the op.
  const doctest::Contains names_op("ops::matmul");
  const Tensor x = Tensor::zeros({2, 64});
  const Tensor w = Tensor::zeros({4, 64}, DType::Q8_0);
  Tensor out = Tensor::zeros({2, 4});

  // w's second dimension is in_features and must match x's width.
  CHECK_THROWS_WITH_AS(ops::matmul(x, Tensor::zeros({4, 32}, DType::Q8_0), out), names_op,
                       std::invalid_argument);
  // out must be [m, n]: wrong row count, wrong column count.
  Tensor out_rows = Tensor::zeros({3, 4});
  CHECK_THROWS_WITH_AS(ops::matmul(x, w, out_rows), names_op, std::invalid_argument);
  Tensor out_cols = Tensor::zeros({2, 5});
  CHECK_THROWS_WITH_AS(ops::matmul(x, w, out_cols), names_op, std::invalid_argument);
  // Every operand must be a matrix.
  CHECK_THROWS_WITH_AS(ops::matmul(Tensor::zeros({2, 1, 64}), w, out), names_op,
                       std::invalid_argument);
  CHECK_THROWS_WITH_AS(ops::matmul(x, Tensor::zeros({64}, DType::Q8_0), out), names_op,
                       std::invalid_argument);
  CHECK_THROWS_WITH_AS(ops::matmul(x, Tensor::zeros({4, 1, 64}, DType::Q8_0), out), names_op,
                       std::invalid_argument);
  Tensor out_cube = Tensor::zeros({2, 4, 1});
  CHECK_THROWS_WITH_AS(ops::matmul(x, w, out_cube), names_op, std::invalid_argument);

  // Only w may be Q8_0: x and out are F32. (A Q8_0 x with an F32 w goes to
  // the F32 kernel, which rejects it too.)
  CHECK_THROWS_WITH_AS(ops::matmul(Tensor::zeros({2, 64}, DType::Q8_0), w, out), names_op,
                       std::invalid_argument);
  CHECK_THROWS_WITH_AS(ops::matmul(Tensor::zeros({2, 64}, DType::BF16), w, out), names_op,
                       std::invalid_argument);
  CHECK_THROWS_WITH_AS(ops::matmul(Tensor::zeros({2, 64}, DType::Q8_0), Tensor::zeros({4, 64}),
                                   out),
                       names_op, std::invalid_argument);
  Tensor out_q8 = Tensor::zeros({2, 32}, DType::Q8_0);
  CHECK_THROWS_WITH_AS(ops::matmul(x, Tensor::zeros({32, 64}, DType::Q8_0), out_q8), names_op,
                       std::invalid_argument);
  Tensor out_bf16 = Tensor::zeros({2, 4}, DType::BF16);
  CHECK_THROWS_WITH_AS(ops::matmul(x, w, out_bf16), names_op, std::invalid_argument);

  // x and out must be contiguous: a column slice keeps its parent's row
  // stride. (A Q8_0 tensor can only be sliced along dimension 0, which keeps
  // it contiguous, so w cannot be made non-contiguous at all.)
  CHECK_THROWS_WITH_AS(ops::matmul(Tensor::zeros({2, 128}).slice(1, 0, 64), w, out), names_op,
                       std::invalid_argument);
  Tensor out_view = Tensor::zeros({2, 8}).slice(1, 0, 4);
  CHECK_THROWS_WITH_AS(ops::matmul(x, w, out_view), names_op, std::invalid_argument);
}

// ---------------------------------------------------------------------------
// BF16 weights (workflow 20). A BF16 w stands for exactly convert_to_f32(w),
// so every case is held to ref::matmul(x, convert_to_f32(w)) with the F32
// kernel's tolerance. m < 24 runs the decode kernel, m >= 24 the panel kernel
// (src/ops/bf16.cpp); the decode kernel takes 4 rows of W at a time and the
// panel kernel 16, so n = 7 and n = 50 leave a tail for both. Unlike Q8_0's,
// a BF16 row has no block structure: the kernels walk k by 8, then 4, then 1,
// so k need not be a multiple of anything.
// ---------------------------------------------------------------------------

TEST_CASE("ops::matmul with a BF16 weight agrees with ref::matmul on convert_to_f32(w)") {
  for (const std::int64_t m : {1, 2, 3, 4, 5, 8, 17, 23, 24, 25, 64, 79}) {
    for (const std::int64_t k : {32, 64, 576, 1536}) {
      for (const std::int64_t n : {1, 7, 50}) {
        const Tensor x = random_tensor({m, k}, 251);
        const Tensor w = random_bf16(n, k, 252);
        check_bf16_matmul(x, w, "ops::matmul bf16 vs ref " + shape_label(m, k, n));
      }
    }
  }
}

TEST_CASE("ops::matmul with a BF16 weight handles every k") {
  // Every tail of the k loops: below one vector, between one and two, and
  // past several, for both kernels.
  for (const std::int64_t m : {1, 2, 3, 4, 5, 8, 17, 23, 24, 29}) {
    for (const std::int64_t k : {1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 17, 31, 33, 577}) {
      for (const std::int64_t n : {1, 5, 25}) {
        const Tensor x = random_tensor({m, k}, 253);
        const Tensor w = random_bf16(n, k, 254);
        check_bf16_matmul(x, w, "ops::matmul bf16 vs ref " + shape_label(m, k, n));
      }
    }
  }
}

TEST_CASE("ops::matmul with a BF16 weight agrees with ref::matmul at the model's shapes") {
  // Every projection shape at m = 1 (decode), m = 64 and m = 79 (the longest
  // oracle prompt), and the full 49152-row lm_head, which only ever sees one
  // row.
  struct Case {
    const char* name;
    std::int64_t k;
    std::int64_t n;
  };
  const Case cases[] = {
      {"q_proj/o_proj", 576, 576},
      {"k_proj/v_proj", 576, 192},
      {"gate_proj/up_proj", 576, 1536},
      {"down_proj", 1536, 576},
  };
  for (const std::int64_t m : {1, 64, 79}) {
    for (const Case& c : cases) {
      const Tensor x = random_tensor({m, c.k}, 255);
      const Tensor w = random_bf16(c.n, c.k, 256);
      check_bf16_matmul(x, w, std::string("ops::matmul bf16 vs ref ") + c.name + " " +
                                  shape_label(m, c.k, c.n));
    }
  }
  const Tensor x = random_tensor({1, 576}, 257);
  check_bf16_matmul(x, random_bf16(49152, 576, 258),
                    "ops::matmul bf16 vs ref lm_head " + shape_label(1, 576, 49152));
}

TEST_CASE("ops::matmul with a BF16 weight agrees with ref::matmul on the checkpoint's weights") {
  // The real model's matrices, which a bf16 checkpoint holds exactly: the
  // first and the last layer's projections at decode and prefill sizes, and
  // the tied lm_head at decode.
  const std::filesystem::path dir = models_dir() / "SmolLM2-135M";
  if (skip_without_checkpoint(dir / "model.safetensors")) return;
  const Weights weights = load_checkpoint(dir);

  const auto check = [&](const std::string& name, std::int64_t m) {
    const Tensor& w32 = weights.get(name);
    const Tensor w = to_bf16(w32);
    const Tensor widened = convert_to_f32(w);
    REQUIRE_MESSAGE(std::memcmp(widened.data(), w32.data(), w32.nbytes()) == 0,
                    name << " is not exactly representable in bf16");
    const Tensor x = random_tensor({m, w.dim(1)}, 259);
    check_bf16_matmul(x, w, "ops::matmul bf16 vs ref " + name + " " +
                                shape_label(m, w.dim(1), w.dim(0)));
  };
  for (const int layer : {0, 29}) {
    const std::string base = "model.layers." + std::to_string(layer) + ".";
    for (const char* proj : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                             "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}) {
      for (const std::int64_t m : {1, 79}) check(base + proj + ".weight", m);
    }
  }
  check("model.embed_tokens.weight", 1);
}

TEST_CASE("ops::matmul with a BF16 weight is bitwise identical at every thread count") {
  // Threads split W into whole tiles (decode) or panels (prefill), and every
  // element is computed the same way whatever the split.
  struct Case {
    const char* name;
    std::int64_t m;
    std::int64_t k;
    std::int64_t n;
  };
  const Case cases[] = {
      {"decode q_proj/o_proj", 1, 576, 576},
      {"decode k_proj/v_proj", 1, 576, 192},
      {"decode gate_proj/up_proj", 1, 576, 1536},
      {"decode down_proj", 1, 1536, 576},
      {"lm_head", 1, 576, 49152},
      {"decode kernel, several rows of x", 3, 576, 576},
      {"decode kernel, a row tail", 23, 576, 192},
      {"prefill gate_proj/up_proj", 64, 576, 1536},
      {"prefill down_proj", 79, 1536, 576},
      {"a tail panel in a threaded split", 25, 64, 1537},
      {"a decode tail in a threaded split", 2, 37, 1537},
      {"the panel kernel at an odd k", 24, 37, 50},
      {"n = 1", 8, 32, 1},
  };
  for (const Case& c : cases) {
    const std::string label =
        std::string("ops::matmul bf16 ") + c.name + " " + shape_label(c.m, c.k, c.n);
    const Tensor x = random_tensor({c.m, c.k}, 260);
    const Tensor w = random_bf16(c.n, c.k, 261);
    const Tensor single = check_thread_invariant(label, [&] {
      Tensor out = nan_filled({c.m, c.n});
      ops::matmul(x, w, out);
      return out;
    });
    Tensor expected = Tensor::zeros({c.m, c.n});
    ref::matmul(x, convert_to_f32(w), expected);
    assert_allclose(single, expected, tol::kMatmul, label + " vs ref");
  }
}

TEST_CASE("ops::matmul with a BF16 weight handles empty dimensions") {
  // k = 0 is an empty sum: every element is 0, even over a NaN-filled out,
  // and x and w may be wrap()s of nullptr.
  for (const std::int64_t m : {1, 8}) {
    Tensor out = nan_filled({m, 4});
    ops::matmul(Tensor::wrap(nullptr, {m, 0}), Tensor::wrap(nullptr, {4, 0}, DType::BF16), out);
    for (std::int64_t i = 0; i < out.numel(); ++i) CHECK(out.f32()[i] == 0.0f);
  }

  // No rows of x, or no rows of W: nothing to write, nothing to read.
  Tensor no_rows = Tensor::zeros({0, 4});
  ops::matmul(Tensor::zeros({0, 64}), random_bf16(4, 64, 262), no_rows);
  Tensor no_cols = Tensor::zeros({3, 0});
  ops::matmul(random_tensor({3, 64}, 263), Tensor::wrap(nullptr, {0, 64}, DType::BF16), no_cols);
}

TEST_CASE("ops::matmul with a BF16 weight rejects invalid arguments") {
  // The same rules as for an F32 w, with errors that name the op.
  const doctest::Contains names_op("ops::matmul");
  const Tensor x = Tensor::zeros({2, 64});
  const Tensor w = Tensor::zeros({4, 64}, DType::BF16);
  Tensor out = Tensor::zeros({2, 4});

  // w's second dimension is in_features and must match x's width.
  CHECK_THROWS_WITH_AS(ops::matmul(x, Tensor::zeros({4, 63}, DType::BF16), out), names_op,
                       std::invalid_argument);
  // out must be [m, n]: wrong row count, wrong column count.
  Tensor out_rows = Tensor::zeros({3, 4});
  CHECK_THROWS_WITH_AS(ops::matmul(x, w, out_rows), names_op, std::invalid_argument);
  Tensor out_cols = Tensor::zeros({2, 5});
  CHECK_THROWS_WITH_AS(ops::matmul(x, w, out_cols), names_op, std::invalid_argument);
  // Every operand must be a matrix.
  CHECK_THROWS_WITH_AS(ops::matmul(Tensor::zeros({2, 1, 64}), w, out), names_op,
                       std::invalid_argument);
  CHECK_THROWS_WITH_AS(ops::matmul(x, Tensor::zeros({64}, DType::BF16), out), names_op,
                       std::invalid_argument);
  CHECK_THROWS_WITH_AS(ops::matmul(x, Tensor::zeros({4, 1, 64}, DType::BF16), out), names_op,
                       std::invalid_argument);
  Tensor out_cube = Tensor::zeros({2, 4, 1});
  CHECK_THROWS_WITH_AS(ops::matmul(x, w, out_cube), names_op, std::invalid_argument);

  // Only w may be BF16: x and out are F32.
  CHECK_THROWS_WITH_AS(ops::matmul(Tensor::zeros({2, 64}, DType::BF16), w, out), names_op,
                       std::invalid_argument);
  CHECK_THROWS_WITH_AS(ops::matmul(Tensor::zeros({2, 64}, DType::Q8_0), w, out), names_op,
                       std::invalid_argument);
  Tensor out_bf16 = Tensor::zeros({2, 4}, DType::BF16);
  CHECK_THROWS_WITH_AS(ops::matmul(x, w, out_bf16), names_op, std::invalid_argument);
  Tensor out_q8 = Tensor::zeros({2, 32}, DType::Q8_0);
  CHECK_THROWS_WITH_AS(ops::matmul(x, Tensor::zeros({32, 64}, DType::BF16), out_q8), names_op,
                       std::invalid_argument);

  // Every operand must be contiguous: a column slice keeps its parent's row
  // stride. Unlike a Q8_0 one, a BF16 w can be sliced that way.
  const Tensor w_view = Tensor::zeros({4, 128}, DType::BF16).slice(1, 0, 64);
  REQUIRE_FALSE(w_view.is_contiguous());
  CHECK_THROWS_WITH_AS(ops::matmul(x, w_view, out), names_op, std::invalid_argument);
  CHECK_THROWS_WITH_AS(ops::matmul(Tensor::zeros({2, 128}).slice(1, 0, 64), w, out), names_op,
                       std::invalid_argument);
  Tensor out_view = Tensor::zeros({2, 8}).slice(1, 0, 4);
  CHECK_THROWS_WITH_AS(ops::matmul(x, w, out_view), names_op, std::invalid_argument);
}
