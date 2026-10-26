// Workflow 03-norm-elementwise owns this file.
//
// In Phase 0 ops:: forwards to ref::, so these pass trivially. That is the
// point: the harness is already here, and the day you replace a body in
// src/ops/norm.cpp or src/ops/elementwise.cpp these tests start doing work.
// Add cases, do not rewrite the structure.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <random>
#include <set>
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

/// An owning Q8_0 [rows, width] table built straight from blocks: every q
/// uniform in [-127, 127], every scale a random finite f16 of either sign,
/// zeros, subnormals and the largest value included. Embedding is exact, so
/// the whole range is fair game.
Tensor random_q8_0_table(std::int64_t rows, std::int64_t width, std::uint64_t seed) {
  Tensor t = Tensor::empty({rows, width}, DType::Q8_0);
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> q(-127, 127);
  std::uniform_int_distribution<int> exponent(0, 30);  // 31 is inf / NaN
  std::uniform_int_distribution<int> mantissa(0, 1023);
  std::uniform_int_distribution<int> sign(0, 1);
  BlockQ8_0* blocks = t.data_as<BlockQ8_0>();
  for (std::int64_t b = 0; b < rows * width / kQ8_0Block; ++b) {
    blocks[b].d = static_cast<std::uint16_t>((sign(rng) << 15) | (exponent(rng) << 10) | mantissa(rng));
    for (std::int8_t& v : blocks[b].qs) v = static_cast<std::int8_t>(q(rng));
  }
  return t;
}

/// An owning BF16 [rows, width] table of uniformly random bit patterns:
/// zeros, subnormals, infinities and NaNs (quiet and signalling) included.
/// Embedding is exact, so the whole range is fair game.
Tensor random_bf16_table(std::int64_t rows, std::int64_t width, std::uint64_t seed) {
  Tensor t = Tensor::empty({rows, width}, DType::BF16);
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> bits(0, 0xffff);
  std::uint16_t* p = t.data_as<std::uint16_t>();
  for (std::int64_t i = 0; i < t.numel(); ++i) p[i] = static_cast<std::uint16_t>(bits(rng));
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
  // A NaN anywhere is the answer; an infinity is an ordinary value. Greedy
  // decoding must not depend on which namespace it calls.
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

TEST_CASE("ops::argmax returns the first NaN, as torch.argmax does") {
  // torch.argmax counts a NaN as greater than every number, +inf included.
  // Each expected index is torch 2.14.1's answer for the same row.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  struct Case {
    const char* name;
    std::vector<float> x;
    std::int32_t expected;
  };
  const Case cases[] = {
      {"NaN first", {nan, 1.0f, 3.0f, 2.0f}, 0},
      {"NaN in the middle", {1.0f, 3.0f, nan, 2.0f}, 2},
      {"NaN last", {1.0f, 3.0f, 2.0f, nan}, 3},
      {"several NaNs", {1.0f, nan, 3.0f, nan, 2.0f, nan}, 1},
      {"+inf before the NaN", {1.0f, inf, nan, 2.0f}, 2},
      {"+inf after the NaN", {1.0f, nan, inf, 2.0f}, 1},
      {"NaN among -inf", {-inf, -inf, nan, -inf}, 2},
      {"all NaN", {nan, nan, nan, nan}, 0},
      {"a lone NaN", {nan}, 0},
  };
  for (const Case& c : cases) {
    Tensor x = Tensor::empty({static_cast<std::int64_t>(c.x.size())});
    std::copy(c.x.begin(), c.x.end(), x.f32());
    CHECK_MESSAGE(ops::argmax(x) == c.expected, c.name);
  }
  CHECK(ops::argmax(nan_filled({kVocab})) == 0);

  // No NaN, but a row whose sum is NaN anyway — inf + -inf, or finite values
  // that overflow both ways — which the kernel's NaN check must see through.
  Tensor infinities = random_tensor({kVocab}, 131);
  infinities.f32()[100] = -inf;
  infinities.f32()[30000] = inf;
  CHECK(ops::argmax(infinities) == 30000);
  const float huge = std::numeric_limits<float>::max();
  Tensor overflowing = Tensor::empty({kVocab + 3});
  for (std::int64_t i = 0; i < kVocab + 3; ++i) overflowing.f32()[i] = i % 16 < 8 ? -huge : huge;
  CHECK(ops::argmax(overflowing) == 8);
  CHECK(ref::argmax(overflowing) == 8);

  // The first NaN in each part of the kernel: either side of the last
  // multiple of 32, 16 and 4, where its main loop, its 16-float blocks and
  // its 4-lane loop end, plus the ends and the middle. Alone, then with more
  // NaNs after it and +inf right before and right after it.
  std::vector<std::int64_t> widths(std::begin(kAwkwardWidths), std::end(kAwkwardWidths));
  widths.push_back(kVocab);
  widths.push_back(kVocab + 3);
  for (const std::int64_t k : widths) {
    std::set<std::int64_t> positions = {0, k / 2, k - 1};
    for (const std::int64_t step : {32, 16, 4}) {
      positions.insert(k / step * step - 1);
      positions.insert(k / step * step);
    }
    const Tensor base = random_tensor({k}, 130);
    for (const std::int64_t first : positions) {
      if (first < 0 || first >= k) continue;
      Tensor alone = base.copy_contiguous();
      alone.f32()[first] = nan;
      Tensor crowded = alone.copy_contiguous();
      for (const std::int64_t later : positions) {
        if (later > first && later < k) crowded.f32()[later] = nan;
      }
      if (first > 0) crowded.f32()[first - 1] = inf;
      if (first + 1 < k && !std::isnan(crowded.f32()[first + 1])) crowded.f32()[first + 1] = inf;
      for (const Tensor* x : {&alone, &crowded}) {
        const char* what = x == &alone ? "alone" : "with more NaNs and +inf";
        CHECK_MESSAGE(ops::argmax(*x) == first,
                      "ops::argmax width " << k << ", first NaN at " << first << ", " << what);
        CHECK_MESSAGE(ref::argmax(*x) == first,
                      "ref::argmax width " << k << ", first NaN at " << first << ", " << what);
      }
    }
  }
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
  // The table may also be Q8_0 or BF16, the weight formats (their cases are
  // below), but no other dtype.
  CHECK_THROWS_AS(ops::embedding(Tensor::zeros({4, 5}, DType::F16), std::vector<std::int32_t>{0},
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

// ---------------------------------------------------------------------------
// Q8_0 embedding tables (workflow 16). A row is dequantized, q * d, which is
// exact in float, so the result must be bit for bit
// ref::embedding(convert_to_f32(table), ...).
// ---------------------------------------------------------------------------

TEST_CASE("ops::embedding with a Q8_0 table is bit for bit ref:: on convert_to_f32(table)") {
  for (const std::int64_t width : {std::int64_t{32}, std::int64_t{64}, kHidden}) {
    const std::string at = " [50, " + std::to_string(width) + "]";
    const Tensor table = random_q8_0_table(50, width, 140);
    const Tensor reference = convert_to_f32(table);

    // The first and last rows, repeats (adjacent and not), and the rest at random.
    std::vector<std::int32_t> ids{0, 49, 7, 7, 0, 49, 23};
    for (const std::int32_t id : random_ids(25, 50, 141)) ids.push_back(id);

    Tensor expected = Tensor::zeros({static_cast<std::int64_t>(ids.size()), width});
    Tensor actual = nan_filled({static_cast<std::int64_t>(ids.size()), width});
    ref::embedding(reference, ids, expected);
    ops::embedding(table, ids, actual);
    assert_identical(actual, expected, "ops::embedding q8_0 vs ref" + at);
  }

  // One block at each extreme of the format: q = +-127 and 0 under scales of
  // +-0, the smallest subnormal, the largest normal and a negative one.
  Tensor table = Tensor::zeros({5, 32}, DType::Q8_0);
  const std::uint16_t scales[] = {0x0000, 0x8000, 0x0001, 0x7bff, 0xbc00};
  BlockQ8_0* blocks = table.data_as<BlockQ8_0>();
  for (int r = 0; r < 5; ++r) {
    blocks[r].d = scales[r];
    for (int j = 0; j < 32; ++j) blocks[r].qs[j] = static_cast<std::int8_t>(j % 3 == 0 ? 127 : j % 3 == 1 ? -127 : 0);
  }
  const std::vector<std::int32_t> ids{0, 1, 2, 3, 4};
  Tensor expected = Tensor::zeros({5, 32});
  Tensor actual = nan_filled({5, 32});
  ref::embedding(convert_to_f32(table), ids, expected);
  ops::embedding(table, ids, actual);
  assert_identical(actual, expected, "ops::embedding q8_0 extreme blocks");
  CHECK(actual.f32()[3 * 32] == 127.0f * 65504.0f);
  CHECK(std::signbit(actual.f32()[1 * 32 + 2]));  // 0 * -0 is -0
}

TEST_CASE("ops::embedding with a Q8_0 table validates its arguments") {
  const doctest::Contains names_op("ops::embedding");
  const Tensor table = random_q8_0_table(4, 64, 142);
  Tensor one = Tensor::zeros({1, 64});

  // Ids out of range throw std::out_of_range, before anything is written.
  Tensor untouched = nan_filled({2, 64});
  CHECK_THROWS_AS(ops::embedding(table, std::vector<std::int32_t>{0, 4}, untouched),
                  std::out_of_range);
  CHECK(std::isnan(untouched.f32()[0]));
  CHECK_THROWS_AS(ops::embedding(table, std::vector<std::int32_t>{-1}, one), std::out_of_range);

  // out must be [n_tokens, width], F32, contiguous.
  CHECK_THROWS_WITH_AS(ops::embedding(table, std::vector<std::int32_t>{0, 1}, one), names_op,
                       std::invalid_argument);
  Tensor narrow = Tensor::zeros({1, 32});
  CHECK_THROWS_WITH_AS(ops::embedding(table, std::vector<std::int32_t>{0}, narrow), names_op,
                       std::invalid_argument);
  Tensor out_q8 = Tensor::zeros({1, 64}, DType::Q8_0);
  CHECK_THROWS_WITH_AS(ops::embedding(table, std::vector<std::int32_t>{0}, out_q8), names_op,
                       std::invalid_argument);
  Tensor out_bf16 = Tensor::zeros({1, 64}, DType::BF16);
  CHECK_THROWS_WITH_AS(ops::embedding(table, std::vector<std::int32_t>{0}, out_bf16), names_op,
                       std::invalid_argument);
  Tensor out_view = Tensor::zeros({2, 128}).slice(1, 0, 64);
  REQUIRE_FALSE(out_view.is_contiguous());
  CHECK_THROWS_WITH_AS(ops::embedding(table, std::vector<std::int32_t>{0, 1}, out_view), names_op,
                       std::invalid_argument);
  Tensor out_rank1 = Tensor::zeros({64});
  CHECK_THROWS_WITH_AS(ops::embedding(table, std::vector<std::int32_t>{0}, out_rank1), names_op,
                       std::invalid_argument);

  // The table must be a matrix.
  CHECK_THROWS_WITH_AS(ops::embedding(Tensor::zeros({64}, DType::Q8_0),
                                      std::vector<std::int32_t>{0}, one),
                       names_op, std::invalid_argument);
  CHECK_THROWS_WITH_AS(ops::embedding(Tensor::zeros({4, 1, 64}, DType::Q8_0),
                                      std::vector<std::int32_t>{0}, one),
                       names_op, std::invalid_argument);
}

TEST_CASE("ops::embedding with a Q8_0 table accepts zero-size inputs") {
  // A wrap() of nullptr is a legal zero-width table; nothing may dereference it.
  const Tensor null_table = Tensor::wrap(nullptr, {4, 0}, DType::Q8_0);
  Tensor null_out = Tensor::wrap(nullptr, {1, 0});
  ops::embedding(null_table, std::vector<std::int32_t>{3}, null_out);

  Tensor no_ids = Tensor::zeros({0, kHidden});
  ops::embedding(random_q8_0_table(4, kHidden, 143), std::span<const std::int32_t>{}, no_ids);
}

// ---------------------------------------------------------------------------
// BF16 embedding tables (workflow 20). A row is widened, which is exact, so
// the result must be bit for bit ref::embedding(convert_to_f32(table), ...),
// NaN payloads included. The kernel widens 8 values at a time, then one.
// ---------------------------------------------------------------------------

TEST_CASE("ops::embedding with a BF16 table is bit for bit ref:: on convert_to_f32(table)") {
  std::vector<std::int64_t> widths(std::begin(kAwkwardWidths), std::end(kAwkwardWidths));
  for (const std::int64_t width : {std::int64_t{8}, std::int64_t{9}, std::int64_t{32}, kHidden}) {
    widths.push_back(width);
  }
  for (const std::int64_t width : widths) {
    const std::string at = " [50, " + std::to_string(width) + "]";
    const Tensor table = random_bf16_table(50, width, 150);
    const Tensor reference = convert_to_f32(table);

    // The first and last rows, repeats (adjacent and not), and the rest at random.
    std::vector<std::int32_t> ids{0, 49, 7, 7, 0, 49, 23};
    for (const std::int32_t id : random_ids(25, 50, 151)) ids.push_back(id);

    Tensor expected = Tensor::zeros({static_cast<std::int64_t>(ids.size()), width});
    Tensor actual = nan_filled({static_cast<std::int64_t>(ids.size()), width});
    ref::embedding(reference, ids, expected);
    ops::embedding(table, ids, actual);
    assert_identical(actual, expected, "ops::embedding bf16 vs ref" + at);
  }

  // The model's whole vocabulary, first and last ids included.
  const Tensor table = random_bf16_table(kVocab, kHidden, 152);
  std::vector<std::int32_t> ids{0, static_cast<std::int32_t>(kVocab - 1), 0};
  for (const std::int32_t id : random_ids(13, static_cast<std::int32_t>(kVocab), 153)) {
    ids.push_back(id);
  }
  Tensor expected = Tensor::zeros({static_cast<std::int64_t>(ids.size()), kHidden});
  Tensor actual = nan_filled({static_cast<std::int64_t>(ids.size()), kHidden});
  ref::embedding(convert_to_f32(table), ids, expected);
  ops::embedding(table, ids, actual);
  assert_identical(actual, expected, "ops::embedding bf16 vs ref, the model's table");
}

TEST_CASE("ops::embedding with a BF16 table widens every class of value exactly") {
  // One row of named bit patterns, checked against the floats they stand for.
  const std::uint16_t bits[] = {
      0x0000,  // +0
      0x8000,  // -0
      0x0001,  // the smallest subnormal
      0x807f,  // the largest negative subnormal
      0x0080,  // the smallest normal
      0x3f80,  // 1
      0xbfc0,  // -1.5
      0x7f7f,  // the largest finite
      0x7f80,  // +inf
      0xff80,  // -inf
      0x7fc0,  // the quiet NaN
      0x7f81,  // a signalling NaN
      0xffff,  // a negative NaN with every payload bit set
  };
  constexpr std::int64_t kWidth = static_cast<std::int64_t>(std::size(bits));
  Tensor table = Tensor::empty({1, kWidth}, DType::BF16);
  std::memcpy(table.data(), bits, sizeof(bits));
  Tensor actual = nan_filled({1, kWidth});
  ops::embedding(table, std::vector<std::int32_t>{0}, actual);
  for (std::int64_t j = 0; j < kWidth; ++j) {
    std::uint32_t got;
    std::memcpy(&got, actual.f32() + j, sizeof(got));
    CHECK_MESSAGE(got == static_cast<std::uint32_t>(bits[j]) << 16, "element " << j);
  }
  CHECK(actual.f32()[5] == 1.0f);
  CHECK(actual.f32()[6] == -1.5f);
  CHECK(actual.f32()[7] == 0x1.fep+127f);  // (2 - 2^-7) * 2^127
  CHECK(std::signbit(actual.f32()[1]));
}

TEST_CASE("ops::embedding with a BF16 table validates its arguments") {
  const doctest::Contains names_op("ops::embedding");
  const Tensor table = random_bf16_table(4, 64, 154);
  Tensor one = Tensor::zeros({1, 64});

  // Ids out of range throw std::out_of_range, before anything is written.
  Tensor untouched = nan_filled({2, 64});
  CHECK_THROWS_AS(ops::embedding(table, std::vector<std::int32_t>{0, 4}, untouched),
                  std::out_of_range);
  CHECK(std::isnan(untouched.f32()[0]));
  CHECK_THROWS_AS(ops::embedding(table, std::vector<std::int32_t>{-1}, one), std::out_of_range);

  // out must be [n_tokens, width], F32, contiguous.
  CHECK_THROWS_WITH_AS(ops::embedding(table, std::vector<std::int32_t>{0, 1}, one), names_op,
                       std::invalid_argument);
  Tensor narrow = Tensor::zeros({1, 32});
  CHECK_THROWS_WITH_AS(ops::embedding(table, std::vector<std::int32_t>{0}, narrow), names_op,
                       std::invalid_argument);
  Tensor out_bf16 = Tensor::zeros({1, 64}, DType::BF16);
  CHECK_THROWS_WITH_AS(ops::embedding(table, std::vector<std::int32_t>{0}, out_bf16), names_op,
                       std::invalid_argument);
  Tensor out_q8 = Tensor::zeros({1, 64}, DType::Q8_0);
  CHECK_THROWS_WITH_AS(ops::embedding(table, std::vector<std::int32_t>{0}, out_q8), names_op,
                       std::invalid_argument);
  Tensor out_view = Tensor::zeros({2, 128}).slice(1, 0, 64);
  REQUIRE_FALSE(out_view.is_contiguous());
  CHECK_THROWS_WITH_AS(ops::embedding(table, std::vector<std::int32_t>{0, 1}, out_view), names_op,
                       std::invalid_argument);
  Tensor out_rank1 = Tensor::zeros({64});
  CHECK_THROWS_WITH_AS(ops::embedding(table, std::vector<std::int32_t>{0}, out_rank1), names_op,
                       std::invalid_argument);

  // The table must be a contiguous matrix. Unlike a Q8_0 one, a BF16 table
  // can be sliced along its rows' width.
  CHECK_THROWS_WITH_AS(ops::embedding(Tensor::zeros({64}, DType::BF16),
                                      std::vector<std::int32_t>{0}, one),
                       names_op, std::invalid_argument);
  CHECK_THROWS_WITH_AS(ops::embedding(Tensor::zeros({4, 1, 64}, DType::BF16),
                                      std::vector<std::int32_t>{0}, one),
                       names_op, std::invalid_argument);
  const Tensor table_view = random_bf16_table(4, 128, 155).slice(1, 0, 64);
  REQUIRE_FALSE(table_view.is_contiguous());
  CHECK_THROWS_WITH_AS(ops::embedding(table_view, std::vector<std::int32_t>{0}, one), names_op,
                       std::invalid_argument);
}

TEST_CASE("ops::embedding with a BF16 table accepts zero-size inputs") {
  // A wrap() of nullptr is a legal zero-width table; nothing may dereference it.
  const Tensor null_table = Tensor::wrap(nullptr, {4, 0}, DType::BF16);
  Tensor null_out = Tensor::wrap(nullptr, {1, 0});
  ops::embedding(null_table, std::vector<std::int32_t>{3}, null_out);

  Tensor no_ids = Tensor::zeros({0, kHidden});
  ops::embedding(random_bf16_table(4, kHidden, 156), std::span<const std::int32_t>{}, no_ids);
}
