// Reference implementations: embedding, silu_mul, add, argmax.
//
// FROZEN ORACLE. These are deliberately the dumbest correct loops that can be
// written. Do not vectorize, parallelize, fuse, or otherwise improve anything
// in src/ops_ref/ — its only job is to be obviously right.

#include <cmath>
#include <stdexcept>

#include "check.h"
#include "minillm/ops.h"

namespace minillm::ref {

using detail::require_dim;
using detail::require_f32_contiguous_rank;
using detail::require_same_shape;

void embedding(const Tensor& table, std::span<const std::int32_t> ids, Tensor& out) {
  require_f32_contiguous_rank("embedding", "table", table, 2);
  require_f32_contiguous_rank("embedding", "out", out, 2);

  const std::int64_t n_tokens = static_cast<std::int64_t>(ids.size());
  const std::int64_t vocab_size = table.shape()[0];
  const std::int64_t hidden = table.shape()[1];

  require_dim("embedding", "out", out, 0, n_tokens, "one row per id");
  require_dim("embedding", "out", out, 1, hidden, "table width");

  const float* src = table.f32();
  float* dst = out.f32();

  for (std::int64_t t = 0; t < n_tokens; ++t) {
    const std::int64_t id = ids[static_cast<std::size_t>(t)];
    if (id < 0 || id >= vocab_size) {
      throw std::out_of_range("ref::embedding: token id " + std::to_string(id) +
                              " out of range for vocab size " + std::to_string(vocab_size));
    }
    for (std::int64_t j = 0; j < hidden; ++j) {
      dst[t * hidden + j] = src[id * hidden + j];
    }
  }
}

void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out) {
  detail::require_f32("silu_mul", "gate", gate);
  detail::require_f32("silu_mul", "up", up);
  detail::require_f32("silu_mul", "out", out);
  detail::require_contiguous("silu_mul", "gate", gate);
  detail::require_contiguous("silu_mul", "up", up);
  detail::require_contiguous("silu_mul", "out", out);
  require_same_shape("silu_mul", "gate", gate, "up", up);
  require_same_shape("silu_mul", "gate", gate, "out", out);

  const float* g = gate.f32();
  const float* u = up.f32();
  float* o = out.f32();
  const std::int64_t n = gate.numel();

  for (std::int64_t i = 0; i < n; ++i) {
    // silu(v) = v * sigmoid(v) = v / (1 + exp(-v))
    const float v = g[i];
    const float silu = v / (1.0f + std::exp(-v));
    o[i] = silu * u[i];
  }
}

void add(const Tensor& a, const Tensor& b, Tensor& out) {
  detail::require_f32("add", "a", a);
  detail::require_f32("add", "b", b);
  detail::require_f32("add", "out", out);
  detail::require_contiguous("add", "a", a);
  detail::require_contiguous("add", "b", b);
  detail::require_contiguous("add", "out", out);
  require_same_shape("add", "a", a, "b", b);
  require_same_shape("add", "a", a, "out", out);

  const float* pa = a.f32();
  const float* pb = b.f32();
  float* po = out.f32();
  const std::int64_t n = a.numel();

  for (std::int64_t i = 0; i < n; ++i) {
    po[i] = pa[i] + pb[i];
  }
}

std::int32_t argmax(const Tensor& x) {
  require_f32_contiguous_rank("argmax", "x", x, 1);
  const std::int64_t n = x.numel();
  if (n == 0) detail::fail("argmax", "x is empty");

  const float* p = x.f32();
  std::int64_t best = 0;
  // Strictly greater, so ties resolve to the lowest index like torch.argmax.
  for (std::int64_t i = 1; i < n; ++i) {
    if (p[i] > p[best]) best = i;
  }
  return static_cast<std::int32_t>(best);
}

}  // namespace minillm::ref
