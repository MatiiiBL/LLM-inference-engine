// Reference implementations: rmsnorm, softmax.
//
// FROZEN ORACLE. One reduction per row, in float, in source order.

#include <cmath>

#include "check.h"
#include "minillm/ops.h"

namespace minillm::ref {

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) {
  detail::require_f32_contiguous_rank("rmsnorm", "x", x, 2);
  detail::require_f32_contiguous_rank("rmsnorm", "weight", weight, 1);
  detail::require_f32_contiguous_rank("rmsnorm", "out", out, 2);
  detail::require_same_shape("rmsnorm", "x", x, "out", out);

  const std::int64_t m = x.shape()[0];
  const std::int64_t k = x.shape()[1];
  detail::require_dim("rmsnorm", "weight", weight, 0, k, "x width");

  const float* px = x.f32();
  const float* pw = weight.f32();
  float* po = out.f32();

  for (std::int64_t i = 0; i < m; ++i) {
    // mean of squares, then 1/sqrt(mean + eps) — matching LlamaRMSNorm with a
    // float32 input: variance = x.pow(2).mean(-1), x * rsqrt(variance + eps).
    float sum_sq = 0.0f;
    for (std::int64_t j = 0; j < k; ++j) {
      const float v = px[i * k + j];
      sum_sq += v * v;
    }
    const float mean_sq = sum_sq / static_cast<float>(k);
    const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

    // Reading px[i*k+j] after writing po[i*k+j] is safe when out aliases x
    // exactly: index j is read before it is written, and never again.
    for (std::int64_t j = 0; j < k; ++j) {
      po[i * k + j] = px[i * k + j] * inv_rms * pw[j];
    }
  }
}

void softmax(Tensor& x) {
  detail::require_f32("softmax", "x", x);
  detail::require_contiguous("softmax", "x", x);
  if (x.ndim() != 1 && x.ndim() != 2) {
    detail::fail("softmax", "x must have rank 1 or 2, got " + x.to_string());
  }

  const std::int64_t rows = x.ndim() == 1 ? 1 : x.shape()[0];
  const std::int64_t cols = x.ndim() == 1 ? x.shape()[0] : x.shape()[1];
  float* p = x.f32();

  for (std::int64_t i = 0; i < rows; ++i) {
    float* row = p + i * cols;
    if (cols == 0) continue;

    // Subtracting the row max is part of the contract, not an optimization:
    // without it exp() overflows for logits above ~88.
    float max_v = row[0];
    for (std::int64_t j = 1; j < cols; ++j) {
      if (row[j] > max_v) max_v = row[j];
    }

    float sum = 0.0f;
    for (std::int64_t j = 0; j < cols; ++j) {
      row[j] = std::exp(row[j] - max_v);
      sum += row[j];
    }

    // sum >= 1 always (the max element contributes exp(0) == 1), so there is
    // no division-by-zero case to guard.
    for (std::int64_t j = 0; j < cols; ++j) {
      row[j] /= sum;
    }
  }
}

}  // namespace minillm::ref
