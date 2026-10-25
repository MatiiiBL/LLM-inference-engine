// ops::rope — owned by workflow 05.
//
// Rotary position embeddings in the Hugging Face rotate_half layout: dimension
// i pairs with i + head_dim/2, NOT with i + 1. ops.h has the contract and
// src/ops_ref/rope.cpp has the full note on why inv_freq must be rounded
// exactly the way transformers rounds it (a float32 exponent, a float32 pow, a
// float32 reciprocal — never double-then-narrow).
//
// Unlike the reference, every argument is validated before anything is
// written, so a throw leaves q and k untouched.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "minillm/ops.h"

namespace minillm::ops {
namespace {

[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument("ops::rope: " + message);
}

void validate(const Tensor& t, const char* name, std::size_t n_positions) {
  if (t.dtype() != DType::F32) fail(std::string(name) + " must be f32, got " + t.to_string());
  if (!t.is_contiguous()) {
    fail(std::string(name) + " must be contiguous, got " + t.to_string());
  }
  if (t.ndim() != 3) fail(std::string(name) + " must have rank 3, got " + t.to_string());
  if (t.dim(0) != static_cast<std::int64_t>(n_positions)) {
    fail(std::string(name) + " has " + std::to_string(t.dim(0)) + " tokens but positions has " +
         std::to_string(n_positions));
  }
  if (t.dim(2) % 2 != 0) {
    fail(std::string(name) + " head_dim must be even, got " + std::to_string(t.dim(2)));
  }
}

void rotate_in_place(Tensor& t, std::span<const std::int64_t> positions,
                     const std::vector<float>& inv_freq) {
  const std::int64_t n_tokens = t.dim(0);
  const std::int64_t n_heads = t.dim(1);
  const std::int64_t head_dim = t.dim(2);
  const std::int64_t half = head_dim / 2;

  float* p = t.f32();
  for (std::int64_t tok = 0; tok < n_tokens; ++tok) {
    const float pos = static_cast<float>(positions[static_cast<std::size_t>(tok)]);
    for (std::int64_t h = 0; h < n_heads; ++h) {
      float* head = p + (tok * n_heads + h) * head_dim;
      for (std::int64_t i = 0; i < half; ++i) {
        // A float32 angle, then cos/sin in double rounded to float — the
        // reference's chain exactly.
        const float angle = pos * inv_freq[static_cast<std::size_t>(i)];
        const float cos_a = static_cast<float>(std::cos(static_cast<double>(angle)));
        const float sin_a = static_cast<float>(std::sin(static_cast<double>(angle)));
        const float lo = head[i];
        const float hi = head[i + half];
        head[i] = lo * cos_a - hi * sin_a;
        head[i + half] = hi * cos_a + lo * sin_a;
      }
    }
  }
}

}  // namespace

void rope(Tensor& q, Tensor* k, std::span<const std::int64_t> positions, float theta) {
  if (!(theta > 0.0f)) fail("theta must be positive, got " + std::to_string(theta));
  validate(q, "q", positions.size());
  if (k != nullptr) {
    validate(*k, "k", positions.size());
    if (k->dim(2) != q.dim(2)) {
      fail("q head_dim " + std::to_string(q.dim(2)) + " != k head_dim " +
           std::to_string(k->dim(2)));
    }
  }
  for (std::size_t t = 0; t < positions.size(); ++t) {
    if (positions[t] < 0) {
      fail("negative position " + std::to_string(positions[t]) + " at token " +
           std::to_string(t));
    }
  }

  const std::int64_t head_dim = q.dim(2);
  const std::int64_t half = head_dim / 2;
  std::vector<float> inv_freq(static_cast<std::size_t>(half));
  for (std::int64_t i = 0; i < half; ++i) {
    const float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
    inv_freq[static_cast<std::size_t>(i)] = 1.0f / std::pow(theta, exponent);
  }

  rotate_in_place(q, positions, inv_freq);
  if (k != nullptr) rotate_in_place(*k, positions, inv_freq);
}

}  // namespace minillm::ops
