// Deterministic test inputs.
//
// Header-only on purpose. Every generator takes an explicit seed so a failure
// is reproducible from the test output alone — never use a clock or a global
// RNG in a test.

#ifndef MINILLM_TESTS_COMMON_RANDOM_H
#define MINILLM_TESTS_COMMON_RANDOM_H

#include <cstdint>
#include <random>
#include <vector>

#include "minillm/tensor.h"

namespace minillm::test {

/// Owning F32 tensor filled with N(0, 1) samples.
inline Tensor random_tensor(const Shape& shape, std::uint64_t seed) {
  Tensor t = Tensor::empty(shape, DType::F32);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  float* p = t.f32();
  for (std::int64_t i = 0; i < t.numel(); ++i) p[i] = dist(rng);
  return t;
}

/// Owning F32 tensor filled with U(lo, hi) samples. Useful for rmsnorm
/// weights, which are all near 1 in a real checkpoint.
inline Tensor random_tensor_uniform(const Shape& shape, std::uint64_t seed, float lo, float hi) {
  Tensor t = Tensor::empty(shape, DType::F32);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  float* p = t.f32();
  for (std::int64_t i = 0; i < t.numel(); ++i) p[i] = dist(rng);
  return t;
}

inline std::vector<std::int32_t> random_ids(std::size_t count, std::int32_t vocab_size,
                                           std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::int32_t> dist(0, vocab_size - 1);
  std::vector<std::int32_t> ids(count);
  for (std::int32_t& id : ids) id = dist(rng);
  return ids;
}

}  // namespace minillm::test

#endif  // MINILLM_TESTS_COMMON_RANDOM_H
