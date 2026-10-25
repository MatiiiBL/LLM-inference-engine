// ref::rmsnorm and ref::softmax vs PyTorch.

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

TEST_CASE("ref::rmsnorm matches torch") {
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
    ref::rmsnorm(x, weight, eps[0], actual);
    assert_allclose(actual, expected, tol::kReduction, "rmsnorm " + variant + " eps=" +
                                                           std::to_string(eps[0]));
  }
}

TEST_CASE("ref::softmax matches torch") {
  // "wide" is a vocabulary-sized row — the shape the sampler softmaxes.
  for (const std::string variant : {"small", "model", "wide"}) {
    const std::string base = "ops/softmax_" + variant + "_";
    MINILLM_SKIP_WITHOUT_ORACLE({base + "x.npy", base + "out.npy"});

    Tensor x = oracle_npy(base + "x.npy").to_tensor();
    const Tensor expected = oracle_npy(base + "out.npy").to_tensor();

    ref::softmax(x);
    assert_allclose(x, expected, tol::kReduction, "softmax " + variant);
  }
}

TEST_CASE("ref::rmsnorm matches the scalar formula and supports aliasing") {
  const std::int64_t k = 6;
  Tensor x = random_tensor({2, k}, 21);
  const Tensor weight = random_tensor_uniform({k}, 22, 0.5f, 1.5f);
  const float eps = 1e-5f;

  Tensor out = Tensor::zeros({2, k});
  ref::rmsnorm(x, weight, eps, out);

  for (std::int64_t i = 0; i < 2; ++i) {
    float sum_sq = 0.0f;
    for (std::int64_t j = 0; j < k; ++j) {
      const float v = x.f32()[i * k + j];
      sum_sq += v * v;
    }
    const float inv = 1.0f / std::sqrt(sum_sq / static_cast<float>(k) + eps);
    for (std::int64_t j = 0; j < k; ++j) {
      CHECK(out.f32()[i * k + j] ==
            doctest::Approx(x.f32()[i * k + j] * inv * weight.f32()[j]).epsilon(1e-6));
    }
  }

  // In-place must give the same answer.
  const Tensor before = out.copy_contiguous();
  ref::rmsnorm(x, weight, eps, x);
  for (std::int64_t i = 0; i < x.numel(); ++i) CHECK(x.f32()[i] == before.f32()[i]);
}

TEST_CASE("ref::rmsnorm with unit weights normalizes the RMS to one") {
  const std::int64_t k = 32;
  Tensor x = random_tensor({3, k}, 23);
  Tensor weight = Tensor::zeros({k});
  for (std::int64_t j = 0; j < k; ++j) weight.f32()[j] = 1.0f;

  Tensor out = Tensor::zeros({3, k});
  ref::rmsnorm(x, weight, 0.0f, out);

  for (std::int64_t i = 0; i < 3; ++i) {
    float sum_sq = 0.0f;
    for (std::int64_t j = 0; j < k; ++j) {
      const float v = out.f32()[i * k + j];
      sum_sq += v * v;
    }
    CHECK(std::sqrt(sum_sq / static_cast<float>(k)) == doctest::Approx(1.0f).epsilon(1e-5));
  }
}

TEST_CASE("ref::softmax rows sum to one and ignore a constant shift") {
  Tensor x = random_tensor({4, 9}, 24);
  Tensor shifted = x.copy_contiguous();
  for (std::int64_t i = 0; i < shifted.numel(); ++i) shifted.f32()[i] += 100.0f;

  ref::softmax(x);
  ref::softmax(shifted);

  for (std::int64_t i = 0; i < 4; ++i) {
    float sum = 0.0f;
    for (std::int64_t j = 0; j < 9; ++j) {
      CHECK(x.f32()[i * 9 + j] > 0.0f);
      sum += x.f32()[i * 9 + j];
    }
    CHECK(sum == doctest::Approx(1.0f).epsilon(1e-6));
  }
  assert_allclose(shifted, x, tol::kReduction, "softmax is shift invariant");
}

TEST_CASE("ref::softmax survives extreme logits") {
  // Without the max subtraction this overflows to inf/nan.
  Tensor x = Tensor::zeros({1, 3});
  x.f32()[0] = 1000.0f;
  x.f32()[1] = 999.0f;
  x.f32()[2] = -1000.0f;
  ref::softmax(x);
  CHECK(std::isfinite(x.f32()[0]));
  CHECK(x.f32()[0] == doctest::Approx(1.0f / (1.0f + std::exp(-1.0f))).epsilon(1e-6));
  CHECK(x.f32()[2] == 0.0f);
  CHECK(x.f32()[0] + x.f32()[1] + x.f32()[2] == doctest::Approx(1.0f).epsilon(1e-6));
}

TEST_CASE("ref::softmax accepts rank 1 and rejects rank 3") {
  Tensor v = Tensor::zeros({4});
  for (std::int64_t i = 0; i < 4; ++i) v.f32()[i] = static_cast<float>(i);
  ref::softmax(v);
  float sum = 0.0f;
  for (std::int64_t i = 0; i < 4; ++i) sum += v.f32()[i];
  CHECK(sum == doctest::Approx(1.0f).epsilon(1e-6));

  Tensor cube = Tensor::zeros({2, 2, 2});
  CHECK_THROWS_AS(ref::softmax(cube), std::invalid_argument);
}

TEST_CASE("ref::rmsnorm rejects a weight of the wrong width") {
  const Tensor x = Tensor::zeros({2, 5});
  Tensor out = Tensor::zeros({2, 5});
  CHECK_THROWS_AS(ref::rmsnorm(x, Tensor::zeros({4}), 1e-5f, out), std::invalid_argument);
  CHECK_THROWS_AS(ref::rmsnorm(x, Tensor::zeros({5, 1}), 1e-5f, out), std::invalid_argument);
}
