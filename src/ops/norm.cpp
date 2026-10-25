// ops::rmsnorm, ops::softmax — owned by workflow 03.
//
// Both are one reduction per row followed by an elementwise pass, and at the
// model's widths both are bound by how fast the row streams through, not by
// arithmetic. minillm::ref does each reduction as one sequential float chain;
// these do it with 4-lane vectors and four independent accumulators (see
// f32x4.h). That reassociates the sum, which is why these are judged against
// ref with tol::kReduction rather than tol::kElementwise.

#include <algorithm>
#include <cmath>

#include "f32x4.h"
#include "minillm/ops.h"
#include "minillm/runtime.h"
#include "validate.h"

namespace minillm::ops {

namespace {

using detail::f32x4;
using detail::kBlock;
using detail::kLanes;
using detail::load;
using detail::splat;
using detail::store;

/// Smallest share of an rmsnorm call worth handing to another thread, in
/// floats. Whole rows are split, so each row's sum is one thread's.
constexpr std::int64_t kRmsnormMinFloats = 8192;

/// sum of p[j]^2 over [0, n). Lane l of accumulator a collects the elements
/// at j = 16 i + 4 a + l; the four accumulators are combined pairwise, their
/// lanes pairwise, and the scalar tail is added last.
float sum_of_squares(const float* p, std::int64_t n) {
  f32x4 acc0 = splat(0.0f);
  f32x4 acc1 = acc0;
  f32x4 acc2 = acc0;
  f32x4 acc3 = acc0;
  std::int64_t j = 0;
  for (; j + kBlock <= n; j += kBlock) {
    const f32x4 v0 = load(p + j);
    const f32x4 v1 = load(p + j + kLanes);
    const f32x4 v2 = load(p + j + 2 * kLanes);
    const f32x4 v3 = load(p + j + 3 * kLanes);
    acc0 += v0 * v0;
    acc1 += v1 * v1;
    acc2 += v2 * v2;
    acc3 += v3 * v3;
  }
  for (; j + kLanes <= n; j += kLanes) {
    const f32x4 v = load(p + j);
    acc0 += v * v;
  }
  float sum = detail::hsum((acc0 + acc1) + (acc2 + acc3));
  for (; j < n; ++j) sum += p[j] * p[j];
  return sum;
}

/// row[j] <- exp(row[j] - shift) for j in [0, n), returning the sum of the
/// new values, accumulated in the same lane pattern as sum_of_squares. The
/// scalar tail uses std::exp, as ref does.
float exp_shifted_and_sum(float* row, std::int64_t n, float shift) {
  const f32x4 vshift = splat(shift);
  f32x4 acc0 = splat(0.0f);
  f32x4 acc1 = acc0;
  f32x4 acc2 = acc0;
  f32x4 acc3 = acc0;
  std::int64_t j = 0;
  for (; j + kBlock <= n; j += kBlock) {
    const f32x4 e0 = detail::exp(load(row + j) - vshift);
    const f32x4 e1 = detail::exp(load(row + j + kLanes) - vshift);
    const f32x4 e2 = detail::exp(load(row + j + 2 * kLanes) - vshift);
    const f32x4 e3 = detail::exp(load(row + j + 3 * kLanes) - vshift);
    store(row + j, e0);
    store(row + j + kLanes, e1);
    store(row + j + 2 * kLanes, e2);
    store(row + j + 3 * kLanes, e3);
    acc0 += e0;
    acc1 += e1;
    acc2 += e2;
    acc3 += e3;
  }
  for (; j + kLanes <= n; j += kLanes) {
    const f32x4 e = detail::exp(load(row + j) - vshift);
    store(row + j, e);
    acc0 += e;
  }
  float sum = detail::hsum((acc0 + acc1) + (acc2 + acc3));
  for (; j < n; ++j) {
    row[j] = std::exp(row[j] - shift);
    sum += row[j];
  }
  return sum;
}

}  // namespace

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) {
  detail::require_f32_contiguous_rank("rmsnorm", "x", x, 2);
  detail::require_f32_contiguous_rank("rmsnorm", "weight", weight, 1);
  detail::require_f32_contiguous_rank("rmsnorm", "out", out, 2);
  detail::require_same_shape("rmsnorm", "x", x, "out", out);

  const std::int64_t m = x.shape()[0];
  const std::int64_t k = x.shape()[1];
  detail::require_dim("rmsnorm", "weight", weight, 0, k, "x width");

  const float* pw = weight.f32();

  const std::int64_t grain = (kRmsnormMinFloats + k - 1) / std::max<std::int64_t>(k, 1);
  kernel_parallel_for(m, grain, [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t i = begin; i < end; ++i) {
      const float* px = x.f32() + i * k;
      float* po = out.f32() + i * k;

      // Same float chain as ref and LlamaRMSNorm after the sum: mean, + eps,
      // sqrt, reciprocal.
      const float mean_sq = sum_of_squares(px, k) / static_cast<float>(k);
      const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

      // (x * inv_rms) * w, in ref's order. Every vector is loaded before the
      // store to the same addresses and never read again, so out may alias x.
      const f32x4 vinv = splat(inv_rms);
      std::int64_t j = detail::for_each_vector(
          k, [&](std::int64_t at) { store(po + at, load(px + at) * vinv * load(pw + at)); });
      for (; j < k; ++j) po[j] = px[j] * inv_rms * pw[j];
    }
  });
}

void softmax(Tensor& x) {
  detail::require_f32_contiguous("softmax", "x", x);
  if (x.ndim() != 1 && x.ndim() != 2) {
    detail::fail("softmax", "x must have rank 1 or 2, got " + x.to_string());
  }

  const std::int64_t rows = x.ndim() == 1 ? 1 : x.shape()[0];
  const std::int64_t cols = x.ndim() == 1 ? x.shape()[0] : x.shape()[1];
  if (cols == 0) return;
  float* p = x.f32();

  for (std::int64_t i = 0; i < rows; ++i) {
    float* row = p + i * cols;

    // The max subtraction is part of the contract: without it exp overflows
    // for logits above ~88. max_of returns exactly the value ref's scalar
    // loop finds, so x - max — and hence the exactness of shift invariance —
    // is the same as ref's.
    const float max_v = detail::max_of(row, cols);
    const float sum = exp_shifted_and_sum(row, cols, max_v);

    // sum >= 1 (the max element contributes exp(0) == 1). Divide, as ref
    // does, rather than multiply by a rounded reciprocal.
    const f32x4 vsum = splat(sum);
    std::int64_t j = detail::for_each_vector(
        cols, [&](std::int64_t at) { store(row + at, load(row + at) / vsum); });
    for (; j < cols; ++j) row[j] /= sum;
  }
}

}  // namespace minillm::ops
