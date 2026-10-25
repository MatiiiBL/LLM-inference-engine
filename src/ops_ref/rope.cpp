// Reference implementation: rope (Hugging Face rotate_half convention).
//
// FROZEN ORACLE.
//
// The arithmetic deliberately mirrors transformers' LlamaRotaryEmbedding step
// for step, because this op is unusually sensitive to how inv_freq is rounded:
//
//   * inv_freq uses a float32 exponent, a float32 pow and a float32
//     reciprocal — the same chain as
//       1.0 / (base ** (arange(0, dim, 2).float() / dim))
//     Computing it in double and narrowing afterwards is MORE accurate but
//     differs from torch by 1 ulp on 11 of the 32 entries, and the angle is
//     position * inv_freq, so a 1-ulp relative difference becomes an absolute
//     angle error of position * 6e-8 radians: 6e-5 at position 4000. cos and
//     sin are 1-Lipschitz, so that lands directly in the output.
//   * the angle is a float32 multiply, exactly as torch computes it
//   * cos/sin are then evaluated in double and rounded to float, which costs
//     nothing here and keeps the table from adding error of its own
//
// Dimension i pairs with i + head_dim/2. NOT the interleaved NeoX layout.

#include <cmath>
#include <vector>

#include "check.h"
#include "minillm/ops.h"

namespace minillm::ref {
namespace {

void rotate_in_place(Tensor& t, const char* name, std::span<const std::int64_t> positions,
                     float theta) {
  detail::require_f32_contiguous_rank("rope", name, t, 3);

  const std::int64_t n_tokens = t.shape()[0];
  const std::int64_t n_heads = t.shape()[1];
  const std::int64_t head_dim = t.shape()[2];

  if (n_tokens != static_cast<std::int64_t>(positions.size())) {
    detail::fail("rope", std::string(name) + " has " + std::to_string(n_tokens) +
                             " tokens but positions has " + std::to_string(positions.size()));
  }
  if (head_dim % 2 != 0) {
    detail::fail("rope", std::string(name) + " head_dim must be even, got " +
                             std::to_string(head_dim));
  }

  const std::int64_t half = head_dim / 2;
  std::vector<float> inv_freq(static_cast<std::size_t>(half));
  for (std::int64_t i = 0; i < half; ++i) {
    // Every step in float32, matching transformers exactly. See the note at
    // the top of this file for why the "more accurate" version is wrong here.
    const float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
    inv_freq[static_cast<std::size_t>(i)] = 1.0f / std::pow(theta, exponent);
  }

  float* p = t.f32();
  for (std::int64_t tok = 0; tok < n_tokens; ++tok) {
    const std::int64_t pos = positions[static_cast<std::size_t>(tok)];
    if (pos < 0) {
      detail::fail("rope", "negative position " + std::to_string(pos) + " at token " +
                               std::to_string(tok));
    }
    for (std::int64_t h = 0; h < n_heads; ++h) {
      float* head = p + (tok * n_heads + h) * head_dim;
      for (std::int64_t i = 0; i < half; ++i) {
        const float angle = static_cast<float>(pos) * inv_freq[static_cast<std::size_t>(i)];
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
  if (!(theta > 0.0f)) {
    detail::fail("rope", "theta must be positive, got " + std::to_string(theta));
  }
  rotate_in_place(q, "q", positions, theta);
  if (k != nullptr) {
    if (k->ndim() == 3 && q.ndim() == 3 && k->shape()[2] != q.shape()[2]) {
      detail::fail("rope", "q head_dim " + std::to_string(q.shape()[2]) + " != k head_dim " +
                               std::to_string(k->shape()[2]));
    }
    rotate_in_place(*k, "k", positions, theta);
  }
}

}  // namespace minillm::ref
