// Reference implementation: matmul (y = x · Wᵀ).
//
// FROZEN ORACLE. Naive triple loop, float accumulator, source order.

#include "check.h"
#include "minillm/ops.h"

namespace minillm::ref {

void matmul(const Tensor& x, const Tensor& w, Tensor& out) {
  detail::require_f32_contiguous_rank("matmul", "x", x, 2);
  detail::require_f32_contiguous_rank("matmul", "w", w, 2);
  detail::require_f32_contiguous_rank("matmul", "out", out, 2);

  const std::int64_t m = x.shape()[0];
  const std::int64_t k = x.shape()[1];
  const std::int64_t n = w.shape()[0];

  // w is [out_features, in_features]: its SECOND dimension matches x's width.
  detail::require_dim("matmul", "w", w, 1, k, "in_features must match x width");
  detail::require_dim("matmul", "out", out, 0, m, "rows of x");
  detail::require_dim("matmul", "out", out, 1, n, "out_features of w");

  const float* px = x.f32();
  const float* pw = w.f32();
  float* po = out.f32();

  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      // Accumulate in float, ascending p. An optimized kernel is free to
      // reassociate this sum; that is the entire reason matmul tolerances are
      // looser than elementwise ones.
      float acc = 0.0f;
      for (std::int64_t p = 0; p < k; ++p) {
        acc += px[i * k + p] * pw[j * k + p];
      }
      po[i * n + j] = acc;
    }
  }
}

}  // namespace minillm::ref
