// ops::embedding, ops::silu_mul, ops::add, ops::argmax — owned by workflow 03.
//
// One pass over the data each, 4 lanes at a time (see f32x4.h), with a scalar
// loop for the tail. add and embedding do exactly ref's arithmetic; silu_mul
// differs from ref only in its exp, which is within 1 ulp of correctly
// rounded; argmax returns exactly ref's index.

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "f32x4.h"
#include "minillm/ops.h"
#include "validate.h"

namespace minillm::ops {

namespace {

using detail::f32x4;
using detail::kBlock;
using detail::kLanes;
using detail::load;
using detail::splat;
using detail::store;

}  // namespace

void embedding(const Tensor& table, std::span<const std::int32_t> ids, Tensor& out) {
  detail::require_f32_contiguous_rank("embedding", "table", table, 2);
  detail::require_f32_contiguous_rank("embedding", "out", out, 2);

  const std::int64_t n_tokens = static_cast<std::int64_t>(ids.size());
  const std::int64_t vocab_size = table.shape()[0];
  const std::int64_t hidden = table.shape()[1];

  detail::require_dim("embedding", "out", out, 0, n_tokens, "one row per id");
  detail::require_dim("embedding", "out", out, 1, hidden, "table width");

  // Every id is checked before anything is written, so a bad one leaves out
  // untouched.
  for (const std::int32_t id : ids) {
    if (id < 0 || id >= vocab_size) {
      throw std::out_of_range("ops::embedding: token id " + std::to_string(id) +
                              " out of range for vocab size " + std::to_string(vocab_size));
    }
  }

  // A zero-width table may be a wrap() of nullptr, and memcpy must not see one.
  const std::size_t row_bytes = static_cast<std::size_t>(hidden) * sizeof(float);
  if (row_bytes == 0) return;

  const float* src = table.f32();
  float* dst = out.f32();
  for (std::int64_t t = 0; t < n_tokens; ++t) {
    const std::int64_t id = ids[static_cast<std::size_t>(t)];
    std::memcpy(dst + t * hidden, src + id * hidden, row_bytes);
  }
}

void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out) {
  detail::require_f32_contiguous("silu_mul", "gate", gate);
  detail::require_f32_contiguous("silu_mul", "up", up);
  detail::require_f32_contiguous("silu_mul", "out", out);
  detail::require_same_shape("silu_mul", "gate", gate, "up", up);
  detail::require_same_shape("silu_mul", "gate", gate, "out", out);

  const float* g = gate.f32();
  const float* u = up.f32();
  float* o = out.f32();
  const std::int64_t n = gate.numel();

  // silu(v) * u = v / (1 + exp(-v)) * u, in ref's order. Both inputs are
  // loaded before the store, so out may alias gate or up.
  const f32x4 one = splat(1.0f);
  std::int64_t i = detail::for_each_vector(n, [&](std::int64_t j) {
    const f32x4 v = load(g + j);
    store(o + j, v / (one + detail::exp(-v)) * load(u + j));
  });
  for (; i < n; ++i) {
    const float v = g[i];
    o[i] = v / (1.0f + std::exp(-v)) * u[i];
  }
}

void add(const Tensor& a, const Tensor& b, Tensor& out) {
  detail::require_f32_contiguous("add", "a", a);
  detail::require_f32_contiguous("add", "b", b);
  detail::require_f32_contiguous("add", "out", out);
  detail::require_same_shape("add", "a", a, "b", b);
  detail::require_same_shape("add", "a", a, "out", out);

  const float* pa = a.f32();
  const float* pb = b.f32();
  float* po = out.f32();
  const std::int64_t n = a.numel();

  std::int64_t i = detail::for_each_vector(
      n, [&](std::int64_t j) { store(po + j, load(pa + j) + load(pb + j)); });
  for (; i < n; ++i) po[i] = pa[i] + pb[i];
}

std::int32_t argmax(const Tensor& x) {
  detail::require_f32_contiguous_rank("argmax", "x", x, 1);
  const std::int64_t n = x.numel();
  if (n == 0) detail::fail("argmax", "x is empty");

  const float* p = x.f32();
  // ref keeps the first strictly greater element, so its answer is the
  // lowest index holding the maximum — except that a NaN in element 0 is
  // never beaten and NaNs elsewhere are never picked. max_of follows the same
  // rule, so the two passes below reproduce ref's index exactly.
  if (std::isnan(p[0])) return 0;
  const float max_v = detail::max_of(p, n);

  // Find the first block holding max_v, then the first element within it.
  // -0.0 == +0.0, so a zero maximum resolves to whichever zero comes first,
  // as in ref.
  const f32x4 vmax = splat(max_v);
  std::int64_t i = 0;
  for (; i + kBlock <= n; i += kBlock) {
    const auto hit = (load(p + i) == vmax) | (load(p + i + kLanes) == vmax) |
                     (load(p + i + 2 * kLanes) == vmax) | (load(p + i + 3 * kLanes) == vmax);
    if (detail::any(hit)) break;
  }
  for (; i < n; ++i) {
    if (p[i] == max_v) return static_cast<std::int32_t>(i);
  }
  // Unreachable: max_v is one of the elements and is not NaN.
  return 0;
}

}  // namespace minillm::ops
