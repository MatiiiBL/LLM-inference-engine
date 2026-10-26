// ops::embedding, ops::silu_mul, ops::add, ops::argmax — owned by workflow 03.
//
// One pass over the data each, 4 lanes at a time (see f32x4.h), with a scalar
// loop for the tail. add and embedding do exactly ref's arithmetic; silu_mul
// differs from ref only in its exp, which is within 1 ulp of correctly
// rounded; argmax returns exactly ref's index.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>

#include "f32x4.h"
#include "minillm/ops.h"
#include "minillm/runtime.h"
#include "q8_0.h"
#include "validate.h"

namespace minillm::ops {

namespace {

using detail::f32x4;
using detail::kBlock;
using detail::kLanes;
using detail::load;
using detail::splat;
using detail::store;

/// Floats per unit of work handed to kernel_parallel_for. A multiple of the
/// vector width, so a chunk boundary never moves an element between the
/// vector loop and the scalar tail: only the end of the whole range has one.
constexpr std::int64_t kUnit = 256;

/// Smallest share of a call worth handing to another thread, in floats.
constexpr std::int64_t kSiluMulMinFloats = 4096;
constexpr std::int64_t kAddMinFloats = 16384;

/// fn(lo, hi) over [0, n) in runs of whole units, at least min_floats long.
template <class Fn>
void parallel_units(std::int64_t n, std::int64_t min_floats, const Fn& fn) {
  const std::int64_t grain = (min_floats + kUnit - 1) / kUnit;
  const auto run = [&](std::int64_t begin, std::int64_t end) {
    fn(begin * kUnit, std::min(end * kUnit, n));
  };
  kernel_parallel_for((n + kUnit - 1) / kUnit, grain, std::ref(run));
}

}  // namespace

void embedding(const Tensor& table, std::span<const std::int32_t> ids, Tensor& out) {
  if (table.dtype() == DType::Q8_0) {
    detail::embedding_q8_0(table, ids, out);
    return;
  }
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

  const std::int64_t n = gate.numel();
  parallel_units(n, kSiluMulMinFloats, [&](std::int64_t lo, std::int64_t hi) {
    const float* g = gate.f32() + lo;
    const float* u = up.f32() + lo;
    float* o = out.f32() + lo;
    const std::int64_t len = hi - lo;

    // silu(v) * u = v / (1 + exp(-v)) * u, in ref's order. Both inputs are
    // loaded before the store, so out may alias gate or up.
    const f32x4 one = splat(1.0f);
    std::int64_t i = detail::for_each_vector(len, [&](std::int64_t j) {
      const f32x4 v = load(g + j);
      store(o + j, v / (one + detail::exp(-v)) * load(u + j));
    });
    for (; i < len; ++i) {
      const float v = g[i];
      o[i] = v / (1.0f + std::exp(-v)) * u[i];
    }
  });
}

void add(const Tensor& a, const Tensor& b, Tensor& out) {
  detail::require_f32_contiguous("add", "a", a);
  detail::require_f32_contiguous("add", "b", b);
  detail::require_f32_contiguous("add", "out", out);
  detail::require_same_shape("add", "a", a, "b", b);
  detail::require_same_shape("add", "a", a, "out", out);

  const std::int64_t n = a.numel();
  parallel_units(n, kAddMinFloats, [&](std::int64_t lo, std::int64_t hi) {
    const float* pa = a.f32() + lo;
    const float* pb = b.f32() + lo;
    float* po = out.f32() + lo;
    const std::int64_t len = hi - lo;
    std::int64_t i = detail::for_each_vector(
        len, [&](std::int64_t j) { store(po + j, load(pa + j) + load(pb + j)); });
    for (; i < len; ++i) po[i] = pa[i] + pb[i];
  });
}

std::int32_t argmax(const Tensor& x) {
  detail::require_f32_contiguous_rank("argmax", "x", x, 1);
  const std::int64_t n = x.numel();
  if (n == 0) detail::fail("argmax", "x is empty");

  const float* p = x.f32();
  // ref's rule, which is torch.argmax's: the first NaN if there is one, else
  // the lowest index holding the maximum.
  //
  // The first pass finds the maximum, NaNs skipped as in max_of, and on the
  // side the sum of every element. A NaN anywhere makes the sum NaN; so can
  // inf + -inf or an overflow, so a NaN sum only means "look": a plain scan
  // then finds the first NaN, or finds none and the maximum stands.
  //
  // Each running maximum is a compare-and-select chain the loop waits on.
  // With max_of's four chains the loop is bound by their latency; with
  // eight, the sum's adds fit in the slack. The maximum does not depend on
  // the order the elements are visited in, up to the sign of a zero
  // maximum, which the == search below ignores.
  constexpr int kChains = 8;
  static_assert(kChains == 8, "the sum below adds eight vectors");
  f32x4 m[kChains];
  for (f32x4& chain : m) chain = splat(p[0]);
  f32x4 sum = splat(0.0f);
  std::int64_t j = 0;
  for (; j + kChains * kLanes <= n; j += kChains * kLanes) {
    f32x4 v[kChains];
    for (int c = 0; c < kChains; ++c) v[c] = load(p + j + c * kLanes);
    for (int c = 0; c < kChains; ++c) m[c] = detail::max_keep(m[c], v[c]);
    sum += ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
  }
  for (; j + kLanes <= n; j += kLanes) {
    const f32x4 v = load(p + j);
    m[0] = detail::max_keep(m[0], v);
    sum += v;
  }
  for (int c = 1; c < kChains; ++c) m[0] = detail::max_keep(m[0], m[c]);
  float max_v = detail::hmax(m[0]);
  float total = detail::hsum(sum);
  for (; j < n; ++j) {
    if (p[j] > max_v) max_v = p[j];
    total += p[j];
  }

  if (std::isnan(total)) {
    for (std::int64_t i = 0; i < n; ++i) {
      if (std::isnan(p[i])) return static_cast<std::int32_t>(i);
    }
  }

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
