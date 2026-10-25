// Workflow 03-norm-elementwise owns this file.
//
// In Phase 0 ops:: forwards to ref::, so these pass trivially. That is the
// point: the harness is already here, and the day you replace a body in
// src/ops/norm.cpp or src/ops/elementwise.cpp these tests start doing work.
// Add cases, do not rewrite the structure.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/allclose.h"
#include "common/oracle.h"
#include "common/random.h"
#include "common/thread_invariant.h"
#include "minillm/ops.h"

using namespace minillm;
using namespace minillm::test;

namespace {

// The real model's shapes, so the kernels are exercised on the widths they
// will actually see. Nothing here is hardcoded into the engine — these are
// test inputs, and the oracle cases below carry the real config values.
constexpr std::int64_t kHidden = 576;
constexpr std::int64_t kIntermediate = 1536;
constexpr std::int64_t kVocab = 49152;

// Widths that are not multiples of the kernels' 4-lane vectors or 16-float
// main loop, plus the boundaries themselves: tail handling is where these
// kernels break. 575 and 577 straddle the hidden size.
constexpr std::int64_t kAwkwardWidths[] = {1, 3, 4, 7, 15, 16, 17, 33, 575, 577};

/// Bitwise equality, for results that must not depend on aliasing or on an
/// exactly representable shift. Reports the first differing index.
void assert_identical(const Tensor& actual, const Tensor& expected, const std::string& label) {
  REQUIRE_MESSAGE(actual.shape() == expected.shape(),
                  label << ": shape " << actual.shape().to_string() << " vs "
                        << expected.shape().to_string());
  const float* a = actual.f32();
  const float* e = expected.f32();
  for (std::int64_t i = 0; i < actual.numel(); ++i) {
    if (std::memcmp(&a[i], &e[i], sizeof(float)) != 0) {
      FAIL_CHECK(label << ": first difference at index " << i << ": " << a[i] << " vs " << e[i]);
      return;
    }
  }
}

Tensor nan_filled(const Shape& shape) {
  Tensor t = Tensor::empty(shape);
  std::fill_n(t.f32(), t.numel(), std::numeric_limits<float>::quiet_NaN());
  return t;
}

}  // namespace

TEST_CASE("ops::rmsnorm agrees with ref::rmsnorm") {
  for (const std::int64_t rows : {1, 8}) {
    const Tensor x = random_tensor({rows, kHidden}, 101);
    const Tensor weight = random_tensor_uniform({kHidden}, 102, 0.5f, 1.5f);
    const float eps = 1e-5f;

    Tensor expected = Tensor::zeros({rows, kHidden});
    Tensor actual = Tensor::zeros({rows, kHidden});
    ref::rmsnorm(x, weight, eps, expected);
    ops::rmsnorm(x, weight, eps, actual);
    assert_allclose(actual, expected, tol::kReduction,
                    "ops::rmsnorm vs ref [" + std::to_string(rows) + ", 576]");
  }
}

TEST_CASE("ops::rmsnorm supports aliasing out onto x") {
  Tensor x = random_tensor({4, kHidden}, 103);
  const Tensor weight = random_tensor_uniform({kHidden}, 104, 0.5f, 1.5f);

  Tensor expected = Tensor::zeros({4, kHidden});
  ref::rmsnorm(x, weight, 1e-5f, expected);
  Tensor unaliased = Tensor::zeros({4, kHidden});
  ops::rmsnorm(x, weight, 1e-5f, unaliased);
  ops::rmsnorm(x, weight, 1e-5f, x);
  assert_allclose(x, expected, tol::kReduction, "ops::rmsnorm in place");
  // Aliasing must not change a single bit, not merely stay within tolerance.
  assert_identical(x, unaliased, "ops::rmsnorm in place vs out of place");

  for (const std::int64_t k : kAwkwardWidths) {
    Tensor y = random_tensor({3, k}, 113);
    const Tensor w = random_tensor_uniform({k}, 114, 0.5f, 1.5f);
    Tensor out = Tensor::zeros({3, k});
    ops::rmsnorm(y, w, 1e-5f, out);
    ops::rmsnorm(y, w, 1e-5f, y);
    assert_identical(y, out, "ops::rmsnorm in place [3, " + std::to_string(k) + "]");
  }
}

TEST_CASE("ops::softmax agrees with ref::softmax") {
  // [9, 128] is an attention-score block; [1, vocab] is what the sampler sees.
  for (const Shape shape : {Shape{9, 128}, Shape{1, kVocab}}) {
    Tensor actual = random_tensor(shape, 105);
    Tensor expected = actual.copy_contiguous();
    ref::softmax(expected);
    ops::softmax(actual);
    assert_allclose(actual, expected, tol::kReduction, "ops::softmax vs ref " + shape.to_string());
  }
}

TEST_CASE("ops::add and ops::silu_mul agree with ref") {
  const Tensor a = random_tensor({8, kHidden}, 106);
  const Tensor b = random_tensor({8, kHidden}, 107);
  Tensor expected_add = Tensor::zeros({8, kHidden});
  Tensor actual_add = Tensor::zeros({8, kHidden});
  ref::add(a, b, expected_add);
  ops::add(a, b, actual_add);
  assert_allclose(actual_add, expected_add, tol::kElementwise, "ops::add vs ref");

  const Tensor gate = random_tensor({8, kIntermediate}, 108);
  const Tensor up = random_tensor({8, kIntermediate}, 109);
  Tensor expected_mul = Tensor::zeros({8, kIntermediate});
  Tensor actual_mul = Tensor::zeros({8, kIntermediate});
  ref::silu_mul(gate, up, expected_mul);
  ops::silu_mul(gate, up, actual_mul);
  assert_allclose(actual_mul, expected_mul, tol::kElementwise, "ops::silu_mul vs ref");
}

TEST_CASE("ops::embedding and ops::argmax agree with ref") {
  const Tensor table = random_tensor({1024, kHidden}, 110);
  const std::vector<std::int32_t> ids = random_ids(8, 1024, 111);
  Tensor expected = Tensor::zeros({8, kHidden});
  Tensor actual = Tensor::zeros({8, kHidden});
  ref::embedding(table, ids, expected);
  ops::embedding(table, ids, actual);
  assert_allclose(actual, expected, tol::kElementwise, "ops::embedding vs ref");

  const Tensor logits = random_tensor({kVocab}, 112);
  CHECK(ops::argmax(logits) == ref::argmax(logits));
}

TEST_CASE("ops:: agrees with ref:: at awkward widths") {
  for (const std::int64_t k : kAwkwardWidths) {
    const std::string at = " [3, " + std::to_string(k) + "]";

    const Tensor x = random_tensor({3, k}, 115);
    const Tensor weight = random_tensor_uniform({k}, 116, 0.5f, 1.5f);
    Tensor expected = Tensor::zeros({3, k});
    Tensor actual = Tensor::zeros({3, k});
    ref::rmsnorm(x, weight, 1e-5f, expected);
    ops::rmsnorm(x, weight, 1e-5f, actual);
    assert_allclose(actual, expected, tol::kReduction, "ops::rmsnorm vs ref" + at);

    // Rank 2 and rank 1: softmax takes both.
    for (const Shape shape : {Shape{3, k}, Shape{k}}) {
      Tensor s_actual = random_tensor(shape, 117);
      Tensor s_expected = s_actual.copy_contiguous();
      ref::softmax(s_expected);
      ops::softmax(s_actual);
      assert_allclose(s_actual, s_expected, tol::kReduction, "ops::softmax vs ref " + shape.to_string());
    }

    const Tensor a = random_tensor({3, k}, 118);
    const Tensor b = random_tensor({3, k}, 119);
    ref::add(a, b, expected);
    ops::add(a, b, actual);
    assert_allclose(actual, expected, tol::kElementwise, "ops::add vs ref" + at);
    ref::silu_mul(a, b, expected);
    ops::silu_mul(a, b, actual);
    assert_allclose(actual, expected, tol::kElementwise, "ops::silu_mul vs ref" + at);

    // The first and last rows of the table, a repeat, and a few in between.
    const Tensor table = random_tensor({16, k}, 120);
    const std::vector<std::int32_t> ids{0, 15, 7, 7, 3};
    Tensor e_expected = Tensor::zeros({5, k});
    Tensor e_actual = Tensor::zeros({5, k});
    ref::embedding(table, ids, e_expected);
    ops::embedding(table, ids, e_actual);
    assert_identical(e_actual, e_expected, "ops::embedding vs ref" + at);

    // The maximum at the head, the middle and the very last (tail) element.
    Tensor logits = random_tensor({k}, 121);
    CHECK(ops::argmax(logits) == ref::argmax(logits));
    for (const std::int64_t pos : {std::int64_t{0}, k / 2, k - 1}) {
      Tensor peaked = logits.copy_contiguous();
      peaked.f32()[pos] = 100.0f;
      CHECK_MESSAGE(ops::argmax(peaked) == pos, "ops::argmax width " << k << " peak at " << pos);
    }
  }
}

TEST_CASE("ops::rmsnorm, ops::silu_mul and ops::add are bitwise identical at every thread count") {
  // Threads split rmsnorm by rows and the elementwise ops by runs of whole
  // vectors, so each element takes exactly its single-thread path, scalar
  // tail included: not close, equal. The shapes are the model's prefill and
  // decode ones, plus sizes that are not a multiple of the vector width and
  // ones large enough to be split at every thread count.
  const Tensor norm_w = random_tensor_uniform({kHidden}, 301, 0.5f, 1.5f);
  for (const std::int64_t rows : {std::int64_t{128}, std::int64_t{1}, std::int64_t{7}}) {
    const std::string label = "ops::rmsnorm [" + std::to_string(rows) + ", 576]";
    const Tensor x = random_tensor({rows, kHidden}, 302);
    const Tensor single = check_thread_invariant(label, [&] {
      Tensor out = nan_filled({rows, kHidden});
      ops::rmsnorm(x, norm_w, 1e-5f, out);
      return out;
    });
    Tensor expected = Tensor::zeros({rows, kHidden});
    ref::rmsnorm(x, norm_w, 1e-5f, expected);
    assert_allclose(single, expected, tol::kReduction, label + " vs ref");
  }
  {
    // 49152 rows: many rows per chunk, at every count.
    const std::string label = "ops::rmsnorm [49152, 7]";
    const Tensor x = random_tensor({kVocab, 7}, 303);
    const Tensor w = random_tensor_uniform({7}, 304, 0.5f, 1.5f);
    const Tensor single = check_thread_invariant(label, [&] {
      Tensor out = nan_filled({kVocab, 7});
      ops::rmsnorm(x, w, 1e-5f, out);
      return out;
    });
    Tensor expected = Tensor::zeros({kVocab, 7});
    ref::rmsnorm(x, w, 1e-5f, expected);
    assert_allclose(single, expected, tol::kReduction, label + " vs ref");
  }

  const Shape shapes[] = {{128, kIntermediate}, {1, kIntermediate}, {128, kHidden}, {1, kHidden},
                          {1},                  {7},                {kVocab},       {kVocab + 3}};
  for (const Shape& shape : shapes) {
    const Tensor a = random_tensor(shape, 305);
    const Tensor b = random_tensor(shape, 306);

    const std::string silu_label = "ops::silu_mul " + shape.to_string();
    const Tensor silu = check_thread_invariant(silu_label, [&] {
      Tensor out = nan_filled(shape);
      ops::silu_mul(a, b, out);
      return out;
    });
    Tensor silu_ref = Tensor::zeros(shape);
    ref::silu_mul(a, b, silu_ref);
    assert_allclose(silu, silu_ref, tol::kElementwise, silu_label + " vs ref");

    const std::string add_label = "ops::add " + shape.to_string();
    const Tensor sum = check_thread_invariant(add_label, [&] {
      Tensor out = nan_filled(shape);
      ops::add(a, b, out);
      return out;
    });
    Tensor sum_ref = Tensor::zeros(shape);
    ref::add(a, b, sum_ref);
    assert_allclose(sum, sum_ref, tol::kElementwise, add_label + " vs ref");
  }
}

TEST_CASE("ops::softmax is exactly shift-invariant") {
  // Multiples of 1/8 below 8 in magnitude: adding 64 is exact, so x - max is
  // bit-identical before and after the shift, and so must the output be.
  for (const std::int64_t cols : {std::int64_t{9}, std::int64_t{128}, std::int64_t{577}}) {
    Tensor x = random_tensor({4, cols}, 122);
    for (std::int64_t i = 0; i < x.numel(); ++i) {
      x.f32()[i] = std::round(x.f32()[i] * 8.0f) / 8.0f;
    }
    Tensor shifted = x.copy_contiguous();
    for (std::int64_t i = 0; i < shifted.numel(); ++i) shifted.f32()[i] += 64.0f;

    ops::softmax(x);
    ops::softmax(shifted);
    assert_identical(shifted, x, "ops::softmax exact shift [4, " + std::to_string(cols) + "]");
  }

  // An inexact shift, as in ref's test: only the rounding of x + 100 differs.
  Tensor x = random_tensor({4, 9}, 123);
  Tensor shifted = x.copy_contiguous();
  for (std::int64_t i = 0; i < shifted.numel(); ++i) shifted.f32()[i] += 100.0f;
  ops::softmax(x);
  ops::softmax(shifted);
  assert_allclose(shifted, x, tol::kReduction, "ops::softmax +100 shift");
}

TEST_CASE("ops::softmax survives extreme logits") {
  // Without the max subtraction this overflows to inf/nan.
  Tensor x = Tensor::zeros({1, 3});
  x.f32()[0] = 1000.0f;
  x.f32()[1] = 999.0f;
  x.f32()[2] = -1000.0f;
  Tensor expected = x.copy_contiguous();
  ref::softmax(expected);
  ops::softmax(x);
  for (std::int64_t i = 0; i < 3; ++i) CHECK(std::isfinite(x.f32()[i]));
  CHECK(x.f32()[2] == 0.0f);
  assert_allclose(x, expected, tol::kReduction, "ops::softmax [1000, 999, -1000]");

  // The same at vocabulary width, with the 1000.0 inside the vector body.
  Tensor wide = random_tensor({kVocab}, 124);
  for (std::int64_t i = 0; i < wide.numel(); ++i) wide.f32()[i] *= 30.0f;
  wide.f32()[12345] = 1000.0f;
  Tensor wide_expected = wide.copy_contiguous();
  ref::softmax(wide_expected);
  ops::softmax(wide);
  for (std::int64_t i = 0; i < wide.numel(); ++i) {
    if (!std::isfinite(wide.f32()[i])) {
      FAIL_CHECK("ops::softmax wide row: non-finite output at index " << i);
      break;
    }
  }
  assert_allclose(wide, wide_expected, tol::kReduction, "ops::softmax wide row with 1000.0");

  // Empty rows are a no-op, as in ref.
  Tensor empty = Tensor::zeros({3, 0});
  ops::softmax(empty);
}

TEST_CASE("ops::add and ops::silu_mul support aliasing out onto an input") {
  for (const std::int64_t k : {kHidden, kIntermediate, std::int64_t{577}}) {
    const std::string at = " [3, " + std::to_string(k) + "]";
    const Tensor a = random_tensor({3, k}, 125);
    const Tensor b = random_tensor({3, k}, 126);
    Tensor unaliased = Tensor::zeros({3, k});

    ops::add(a, b, unaliased);
    Tensor onto_a = a.copy_contiguous();
    ops::add(onto_a, b, onto_a);
    assert_identical(onto_a, unaliased, "ops::add out = a" + at);
    Tensor onto_b = b.copy_contiguous();
    ops::add(a, onto_b, onto_b);
    assert_identical(onto_b, unaliased, "ops::add out = b" + at);

    ops::silu_mul(a, b, unaliased);
    Tensor onto_gate = a.copy_contiguous();
    ops::silu_mul(onto_gate, b, onto_gate);
    assert_identical(onto_gate, unaliased, "ops::silu_mul out = gate" + at);
    Tensor onto_up = b.copy_contiguous();
    ops::silu_mul(a, onto_up, onto_up);
    assert_identical(onto_up, unaliased, "ops::silu_mul out = up" + at);

    // All three the same tensor: out = a + a.
    Tensor all = a.copy_contiguous();
    ops::add(a, a, unaliased);
    ops::add(all, all, all);
    assert_identical(all, unaliased, "ops::add out = a = b" + at);
  }
}

TEST_CASE("ops::argmax resolves ties to the lowest index") {
  Tensor x = Tensor::zeros({5});
  x.f32()[1] = 2.0f;
  x.f32()[3] = 2.0f;
  CHECK(ops::argmax(x) == 1);

  // Every element tied.
  CHECK(ops::argmax(Tensor::zeros({kVocab})) == 0);

  // Ties in the same vector, in different vectors of one block, in different
  // blocks, and between the vector body and the scalar tail.
  Tensor logits = random_tensor({kVocab + 3}, 127);
  for (const std::int64_t pos : {std::int64_t{40000}, kVocab + 2, std::int64_t{21},
                                 std::int64_t{6}, std::int64_t{5}}) {
    logits.f32()[pos] = 100.0f;
  }
  CHECK(ops::argmax(logits) == 5);
  CHECK(ops::argmax(logits) == ref::argmax(logits));

  // -0.0 and +0.0 compare equal, so whichever comes first wins.
  Tensor zeros = Tensor::zeros({3});
  zeros.f32()[0] = -1.0f;
  zeros.f32()[1] = -0.0f;
  CHECK(ops::argmax(zeros) == 1);
  zeros.f32()[1] = 0.0f;
  zeros.f32()[2] = -0.0f;
  CHECK(ops::argmax(zeros) == 1);
}

TEST_CASE("ops::argmax agrees with ref around NaN and infinities") {
  // ref compares with a strict >, so a NaN is never picked unless it is
  // element 0 — which then can never be beaten. Greedy decoding must not
  // depend on which namespace it calls.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (const std::int64_t k : {std::int64_t{1}, std::int64_t{17}, std::int64_t{577}, kVocab}) {
    const Tensor base = random_tensor({k}, 128);
    for (const std::int64_t pos : {std::int64_t{0}, k / 2, k - 1}) {
      for (const float special : {nan, inf, -inf}) {
        Tensor x = base.copy_contiguous();
        x.f32()[pos] = special;
        CHECK_MESSAGE(ops::argmax(x) == ref::argmax(x),
                      "ops::argmax width " << k << ", " << special << " at " << pos);
      }
    }
  }
  Tensor all_neg_inf = Tensor::zeros({33});
  for (std::int64_t i = 0; i < 33; ++i) all_neg_inf.f32()[i] = -inf;
  CHECK(ops::argmax(all_neg_inf) == 0);
}

TEST_CASE("ops::rmsnorm matches torch") {
  for (const std::string variant : {"small", "model"}) {
    const std::string base = "ops/rmsnorm_" + variant + "_";
    MINILLM_SKIP_WITHOUT_ORACLE(
        {base + "x.npy", base + "weight.npy", base + "eps.npy", base + "out.npy"});

    const Tensor x = oracle_npy(base + "x.npy").to_tensor();
    const Tensor weight = oracle_npy(base + "weight.npy").to_tensor();
    const std::vector<float> eps = oracle_npy(base + "eps.npy").as_f32();
    const Tensor expected = oracle_npy(base + "out.npy").to_tensor();
    REQUIRE(eps.size() == 1);

    Tensor actual = Tensor::zeros(expected.shape());
    ops::rmsnorm(x, weight, eps[0], actual);
    assert_allclose(actual, expected, tol::kReduction, "ops::rmsnorm " + variant);
  }
}

TEST_CASE("ops::softmax matches torch") {
  for (const std::string variant : {"small", "model", "wide"}) {
    const std::string base = "ops/softmax_" + variant + "_";
    MINILLM_SKIP_WITHOUT_ORACLE({base + "x.npy", base + "out.npy"});

    Tensor x = oracle_npy(base + "x.npy").to_tensor();
    const Tensor expected = oracle_npy(base + "out.npy").to_tensor();
    ops::softmax(x);
    assert_allclose(x, expected, tol::kReduction, "ops::softmax " + variant);
  }
}

TEST_CASE("ops::add, ops::silu_mul and ops::embedding match torch") {
  for (const std::string variant : {"small", "model"}) {
    {
      const std::string base = "ops/add_" + variant + "_";
      MINILLM_SKIP_WITHOUT_ORACLE({base + "a.npy", base + "b.npy", base + "out.npy"});
      const Tensor a = oracle_npy(base + "a.npy").to_tensor();
      const Tensor b = oracle_npy(base + "b.npy").to_tensor();
      const Tensor expected = oracle_npy(base + "out.npy").to_tensor();
      Tensor actual = Tensor::zeros(expected.shape());
      ops::add(a, b, actual);
      assert_allclose(actual, expected, tol::kElementwise, "ops::add " + variant);
    }
    {
      const std::string base = "ops/silu_mul_" + variant + "_";
      MINILLM_SKIP_WITHOUT_ORACLE({base + "gate.npy", base + "up.npy", base + "out.npy"});
      const Tensor gate = oracle_npy(base + "gate.npy").to_tensor();
      const Tensor up = oracle_npy(base + "up.npy").to_tensor();
      const Tensor expected = oracle_npy(base + "out.npy").to_tensor();
      Tensor actual = Tensor::zeros(expected.shape());
      ops::silu_mul(gate, up, actual);
      assert_allclose(actual, expected, tol::kElementwise, "ops::silu_mul " + variant);
    }
    {
      const std::string base = "ops/embedding_" + variant + "_";
      MINILLM_SKIP_WITHOUT_ORACLE({base + "table.npy", base + "ids.npy", base + "out.npy"});
      const Tensor table = oracle_npy(base + "table.npy").to_tensor();
      const std::vector<std::int32_t> ids = oracle_npy(base + "ids.npy").as_i32_narrowing();
      const Tensor expected = oracle_npy(base + "out.npy").to_tensor();
      Tensor actual = Tensor::zeros(expected.shape());
      ops::embedding(table, ids, actual);
      assert_allclose(actual, expected, tol::kElementwise, "ops::embedding " + variant);
    }
  }
}

TEST_CASE("ops::argmax matches torch") {
  for (const std::string variant : {"small", "model"}) {
    const std::string base = "ops/argmax_" + variant + "_";
    MINILLM_SKIP_WITHOUT_ORACLE({base + "x.npy", base + "out.npy"});

    const Tensor x = oracle_npy(base + "x.npy").to_tensor();
    const std::vector<std::int64_t> expected = oracle_npy(base + "out.npy").as_i64();
    REQUIRE(expected.size() == 1);
    CHECK_MESSAGE(ops::argmax(x) == static_cast<std::int32_t>(expected[0]), "ops::argmax " << variant);
  }
}

TEST_CASE("ops:: kernels validate their arguments") {
  // ops.h: shape validation is not a debug-only check, optimized kernels too.
  const Tensor x = Tensor::zeros({2, 5});
  Tensor out = Tensor::zeros({2, 5});
  const Tensor bf16 = Tensor::zeros({2, 5}, DType::BF16);
  const Tensor strided = Tensor::zeros({2, 10}).slice(1, 0, 5);
  REQUIRE_FALSE(strided.is_contiguous());

  const Tensor weight = Tensor::zeros({5});
  CHECK_THROWS_AS(ops::rmsnorm(x, Tensor::zeros({4}), 1e-5f, out), std::invalid_argument);
  CHECK_THROWS_AS(ops::rmsnorm(x, Tensor::zeros({5, 1}), 1e-5f, out), std::invalid_argument);
  Tensor narrow = Tensor::zeros({2, 4});
  CHECK_THROWS_AS(ops::rmsnorm(x, weight, 1e-5f, narrow), std::invalid_argument);
  CHECK_THROWS_AS(ops::rmsnorm(Tensor::zeros({5}), weight, 1e-5f, out), std::invalid_argument);
  CHECK_THROWS_AS(ops::rmsnorm(strided, weight, 1e-5f, out), std::invalid_argument);
  CHECK_THROWS_AS(ops::rmsnorm(bf16, weight, 1e-5f, out), std::invalid_argument);

  Tensor cube = Tensor::zeros({2, 2, 2});
  CHECK_THROWS_AS(ops::softmax(cube), std::invalid_argument);
  Tensor strided_rows = Tensor::zeros({2, 10}).slice(1, 0, 5);
  CHECK_THROWS_AS(ops::softmax(strided_rows), std::invalid_argument);
  Tensor bf16_rows = Tensor::zeros({2, 5}, DType::BF16);
  CHECK_THROWS_AS(ops::softmax(bf16_rows), std::invalid_argument);

  const Tensor transposed = Tensor::zeros({5, 2});
  for (auto* fn : {&ops::add, &ops::silu_mul}) {
    CHECK_THROWS_AS(fn(x, transposed, out), std::invalid_argument);
    CHECK_THROWS_AS(fn(x, x, narrow), std::invalid_argument);
    CHECK_THROWS_AS(fn(x, bf16, out), std::invalid_argument);
    CHECK_THROWS_AS(fn(strided, x, out), std::invalid_argument);
  }

  const Tensor table = Tensor::zeros({4, 5});
  Tensor one = Tensor::zeros({1, 5});
  CHECK_THROWS_AS(ops::embedding(table, std::vector<std::int32_t>{4}, one), std::out_of_range);
  CHECK_THROWS_AS(ops::embedding(table, std::vector<std::int32_t>{-1}, one), std::out_of_range);
  CHECK_THROWS_AS(ops::embedding(table, std::vector<std::int32_t>{0, 1}, one),
                  std::invalid_argument);
  CHECK_THROWS_AS(ops::embedding(table, std::vector<std::int32_t>{0}, narrow),
                  std::invalid_argument);
  CHECK_THROWS_AS(ops::embedding(Tensor::zeros({4, 5}, DType::BF16), std::vector<std::int32_t>{0},
                                 one),
                  std::invalid_argument);

  CHECK_THROWS_AS(ops::argmax(Tensor::zeros({2, 2})), std::invalid_argument);
  CHECK_THROWS_AS(ops::argmax(Tensor::zeros({0})), std::invalid_argument);
  CHECK_THROWS_AS(ops::argmax(Tensor::zeros({5}, DType::BF16)), std::invalid_argument);
}

TEST_CASE("ops:: kernels accept zero-size inputs") {
  // A wrap() of nullptr is a legal zero-size tensor; nothing may dereference it.
  const Tensor null_table = Tensor::wrap(nullptr, {4, 0});
  Tensor null_out = Tensor::wrap(nullptr, {1, 0});
  ops::embedding(null_table, std::vector<std::int32_t>{3}, null_out);

  Tensor no_ids = Tensor::zeros({0, kHidden});
  ops::embedding(Tensor::zeros({4, kHidden}), std::span<const std::int32_t>{}, no_ids);

  const Tensor no_rows = Tensor::zeros({0, kHidden});
  Tensor no_rows_out = Tensor::zeros({0, kHidden});
  ops::rmsnorm(no_rows, Tensor::zeros({kHidden}), 1e-5f, no_rows_out);
  ops::add(no_rows, no_rows, no_rows_out);
  ops::silu_mul(no_rows, no_rows, no_rows_out);
  Tensor no_cols = Tensor::zeros({0});
  ops::softmax(no_cols);
}
