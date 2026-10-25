// ref::rope vs the RoPE in transformers' Llama modeling code.
//
// The point of these tests is the PAIRING: dimension i with i + head_dim/2.
// The interleaved alternative produces numbers of the same magnitude in the
// same places, so only a golden-data comparison catches it.

#include <doctest/doctest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/allclose.h"
#include "common/oracle.h"
#include "common/random.h"
#include "minillm/ops.h"

using namespace minillm;
using namespace minillm::test;

TEST_CASE("ref::rope matches transformers") {
  // "offset" uses a single token at a large absolute position, which is the
  // decode-step case and the one where a float32 angle loses the most bits.
  for (const std::string variant : {"small", "model", "offset"}) {
    const std::string base = "ops/rope_" + variant + "_";
    MINILLM_SKIP_WITHOUT_ORACLE({base + "q.npy", base + "k.npy", base + "positions.npy",
                                 base + "theta.npy", base + "q_out.npy", base + "k_out.npy"});

    Tensor q = oracle_npy(base + "q.npy").to_tensor();
    Tensor k = oracle_npy(base + "k.npy").to_tensor();
    const std::vector<std::int64_t> positions = oracle_npy(base + "positions.npy").as_i64();
    const std::vector<float> theta = oracle_npy(base + "theta.npy").as_f32();
    const Tensor q_expected = oracle_npy(base + "q_out.npy").to_tensor();
    const Tensor k_expected = oracle_npy(base + "k_out.npy").to_tensor();
    REQUIRE(theta.size() == 1);

    const std::int64_t max_position = positions.empty() ? 0 : positions.back();
    const Tolerance rope_tol = rope_tolerance(max_position);

    ref::rope(q, &k, positions, theta[0]);
    const std::string label = "rope " + variant + " max_position=" + std::to_string(max_position);
    assert_allclose(q, q_expected, rope_tol, label + " q");
    assert_allclose(k, k_expected, rope_tol, label + " k");
  }
}

TEST_CASE("ref::rope pairs dimension i with i + head_dim/2") {
  // One token at position 1, one head, head_dim 4 => half = 2, and
  // inv_freq[0] = 1 so the first pair rotates by exactly 1 radian.
  Tensor q = Tensor::zeros({1, 1, 4});
  q.f32()[0] = 1.0f;  // dimension 0
  q.f32()[1] = 0.0f;  // dimension 1
  q.f32()[2] = 0.0f;  // dimension 2 == 0 + half
  q.f32()[3] = 0.0f;  // dimension 3 == 1 + half

  const std::vector<std::int64_t> positions{1};
  ref::rope(q, nullptr, positions, 10000.0f);

  // Dimension 0 pairs with dimension 2: x0 -> x0*cos - x2*sin, x2 -> x2*cos + x0*sin.
  CHECK(q.f32()[0] == doctest::Approx(std::cos(1.0)).epsilon(1e-6));
  CHECK(q.f32()[2] == doctest::Approx(std::sin(1.0)).epsilon(1e-6));
  // Dimension 1 is untouched because it was zero and its partner is zero.
  CHECK(q.f32()[1] == 0.0f);
  CHECK(q.f32()[3] == 0.0f);

  // The interleaved convention would have put sin(1) in dimension 1.
  CHECK(q.f32()[1] != doctest::Approx(std::sin(1.0)).epsilon(1e-6));
}

TEST_CASE("ref::rope at position 0 is the identity") {
  Tensor q = random_tensor({3, 2, 8}, 31);
  Tensor k = random_tensor({3, 1, 8}, 32);
  const Tensor q_before = q.copy_contiguous();
  const Tensor k_before = k.copy_contiguous();

  const std::vector<std::int64_t> positions{0, 0, 0};
  ref::rope(q, &k, positions, 100000.0f);

  for (std::int64_t i = 0; i < q.numel(); ++i) CHECK(q.f32()[i] == q_before.f32()[i]);
  for (std::int64_t i = 0; i < k.numel(); ++i) CHECK(k.f32()[i] == k_before.f32()[i]);
}

TEST_CASE("ref::rope preserves the norm of each rotated pair") {
  Tensor q = random_tensor({4, 3, 16}, 33);
  const Tensor before = q.copy_contiguous();
  const std::vector<std::int64_t> positions{0, 1, 7, 4095};
  ref::rope(q, nullptr, positions, 100000.0f);

  const std::int64_t head_dim = 16;
  const std::int64_t half = head_dim / 2;
  for (std::int64_t t = 0; t < 4; ++t) {
    for (std::int64_t h = 0; h < 3; ++h) {
      for (std::int64_t i = 0; i < half; ++i) {
        const std::int64_t lo = (t * 3 + h) * head_dim + i;
        const std::int64_t hi = lo + half;
        const double n_before = static_cast<double>(before.f32()[lo]) * before.f32()[lo] +
                                static_cast<double>(before.f32()[hi]) * before.f32()[hi];
        const double n_after = static_cast<double>(q.f32()[lo]) * q.f32()[lo] +
                               static_cast<double>(q.f32()[hi]) * q.f32()[hi];
        CHECK(n_after == doctest::Approx(n_before).epsilon(1e-5));
      }
    }
  }
}

TEST_CASE("ref::rope rotating q alone leaves k untouched") {
  Tensor q = random_tensor({2, 2, 4}, 34);
  Tensor k = random_tensor({2, 1, 4}, 35);
  const Tensor k_before = k.copy_contiguous();

  const std::vector<std::int64_t> positions{3, 4};
  ref::rope(q, nullptr, positions, 10000.0f);
  for (std::int64_t i = 0; i < k.numel(); ++i) CHECK(k.f32()[i] == k_before.f32()[i]);
}

TEST_CASE("ref::rope validates its arguments") {
  Tensor q = Tensor::zeros({2, 2, 4});
  const std::vector<std::int64_t> two{0, 1};

  // positions must have one entry per token.
  CHECK_THROWS_AS(ref::rope(q, nullptr, std::vector<std::int64_t>{0}, 10000.0f),
                  std::invalid_argument);
  // head_dim must be even.
  Tensor odd = Tensor::zeros({2, 2, 5});
  CHECK_THROWS_AS(ref::rope(odd, nullptr, two, 10000.0f), std::invalid_argument);
  // rank must be 3.
  Tensor flat = Tensor::zeros({2, 4});
  CHECK_THROWS_AS(ref::rope(flat, nullptr, two, 10000.0f), std::invalid_argument);
  // theta must be positive.
  CHECK_THROWS_AS(ref::rope(q, nullptr, two, 0.0f), std::invalid_argument);
  // negative positions are a bug, not a convention.
  CHECK_THROWS_AS(ref::rope(q, nullptr, std::vector<std::int64_t>{0, -1}, 10000.0f),
                  std::invalid_argument);
  // q and k must agree on head_dim.
  Tensor k_wrong = Tensor::zeros({2, 1, 8});
  CHECK_THROWS_AS(ref::rope(q, &k_wrong, two, 10000.0f), std::invalid_argument);
}
