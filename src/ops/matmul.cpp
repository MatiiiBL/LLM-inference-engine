// ops::matmul — owned by workflow 04.
//
// out = x · Wᵀ, with W stored [n, k] exactly as the checkpoint has it. Row j
// of W is contiguous, so every output element is a dot product of two
// contiguous rows of length k.

#include <cstdint>
#include <stdexcept>
#include <string>

#include "minillm/ops.h"

namespace minillm::ops {
namespace {

[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument("ops::matmul: " + message);
}

void require_matrix(const char* name, const Tensor& t) {
  if (t.dtype() != DType::F32) fail(std::string(name) + " must be f32, got " + t.to_string());
  if (!t.is_contiguous()) fail(std::string(name) + " must be contiguous, got " + t.to_string());
  if (t.ndim() != 2) fail(std::string(name) + " must have rank 2, got " + t.to_string());
}

void require_dim(const char* name, const Tensor& t, int dim, std::int64_t expected,
                 const char* what) {
  if (t.shape()[dim] != expected) {
    fail(std::string(name) + " dimension " + std::to_string(dim) + " is " +
         std::to_string(t.shape()[dim]) + ", expected " + std::to_string(expected) + " (" +
         what + ")");
  }
}

}  // namespace

void matmul(const Tensor& x, const Tensor& w, Tensor& out) {
  require_matrix("x", x);
  require_matrix("w", w);
  require_matrix("out", out);

  const std::int64_t m = x.shape()[0];
  const std::int64_t k = x.shape()[1];
  const std::int64_t n = w.shape()[0];

  require_dim("w", w, 1, k, "in_features must match x width");
  require_dim("out", out, 0, m, "rows of x");
  require_dim("out", out, 1, n, "out_features of w");

  const float* px = x.f32();
  const float* pw = w.f32();
  float* po = out.f32();

  for (std::int64_t i = 0; i < m; ++i) {
    const float* xi = px + i * k;
    for (std::int64_t j = 0; j < n; ++j) {
      const float* wj = pw + j * k;
      float acc = 0.0f;
      for (std::int64_t p = 0; p < k; ++p) acc += xi[p] * wj[p];
      po[i * n + j] = acc;
    }
  }
}

}  // namespace minillm::ops
