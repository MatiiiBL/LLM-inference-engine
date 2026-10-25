// ops::matmul — owned by workflow 04.
//
// out = x · Wᵀ, with W stored [n, k] exactly as the checkpoint has it. Row j
// of W is contiguous, so every output element is a dot product of two
// contiguous rows of length k.
//
// The kernel is a register tile: MR rows of x against NR rows of W, keeping
// MR * NR four-lane accumulators that run along k, reduced horizontally once
// at the end. Each x vector loaded is reused NR times and each W vector MR
// times. The loop nest walks W one NR-row panel at a time and sweeps every
// row block of x across it, so the panel (NR * k floats, 9-25 KB here) stays
// in L1 and W is streamed from memory exactly once per call.
//
// NR = 4 is a measurement, not a guess. At m = 1 (decode) the kernel is a
// matrix-vector product bound by streaming W, and on an M4 P-core walking 2-4
// W rows at once streams at ~52 GB/s (85% of a plain sequential read) while
// 8 or 16 rows at once drop to ~22 GB/s: the hardware prefetcher stops
// keeping up with that many concurrent streams. MR = 4 makes the m = 8 tile
// 16 accumulators, which covers FMA latency without spilling.
//
// Lanes and accumulator sets reassociate the sum relative to ref::matmul's
// source order; ops.h allows that and tol::kMatmul is sized for it.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "minillm/ops.h"
#include "minillm/runtime.h"

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

// Four float lanes. NEON on AArch64 (where it is always present); elsewhere a
// plain struct with the same operations, which the compiler vectorizes as far
// as the target allows. The kernel below is written once against these.
#if defined(__aarch64__)
using Vec = float32x4_t;
inline Vec vec_zero() { return vdupq_n_f32(0.0f); }
inline Vec vec_load(const float* p) { return vld1q_f32(p); }
inline Vec vec_fma(Vec acc, Vec a, Vec b) { return vfmaq_f32(acc, a, b); }
inline Vec vec_add(Vec a, Vec b) { return vaddq_f32(a, b); }
inline float vec_sum(Vec v) { return vaddvq_f32(v); }
#else
struct Vec {
  float lane[4];
};
inline Vec vec_zero() { return Vec{}; }
inline Vec vec_load(const float* p) { return Vec{{p[0], p[1], p[2], p[3]}}; }
inline Vec vec_fma(Vec acc, Vec a, Vec b) {
  for (int l = 0; l < 4; ++l) acc.lane[l] += a.lane[l] * b.lane[l];
  return acc;
}
inline Vec vec_add(Vec a, Vec b) {
  for (int l = 0; l < 4; ++l) a.lane[l] += b.lane[l];
  return a;
}
inline float vec_sum(Vec v) { return (v.lane[0] + v.lane[1]) + (v.lane[2] + v.lane[3]); }
#endif

constexpr int kRows = 4;  // MR: rows of x per tile
constexpr int kCols = 4;  // NR: rows of W per tile

// Smallest share of a call worth handing to another thread, in multiply-adds.
// A call with less than two of these runs inline.
constexpr std::int64_t kMinChunkMacs = std::int64_t{1} << 13;

// Independent accumulator sets along k. A tile needs ~16 accumulator chains
// in flight to hide FMA latency; the full 4x4 tile has them, the narrower
// tail tiles (and the whole m = 1 decode step, which is all 1x4 tiles) get
// them by splitting k across 2 or 4 sets.
constexpr int accumulator_sets(int tiles) { return tiles >= 16 ? 1 : tiles >= 8 ? 2 : 4; }

// out[i, j] for i < MR, j < NR, where x points at row 0 of the tile's x rows,
// w at row 0 of its W rows, and out at element [0, 0]. k is the row length
// of x and w; n is the row length of out.
template <int MR, int NR>
void tile(const float* x, const float* w, float* out, std::int64_t k, std::int64_t n) {
  constexpr int KU = accumulator_sets(MR * NR);
  Vec acc[KU][MR][NR];
  for (int u = 0; u < KU; ++u)
    for (int i = 0; i < MR; ++i)
      for (int j = 0; j < NR; ++j) acc[u][i][j] = vec_zero();

  std::int64_t p = 0;
  for (; p + 4 * KU <= k; p += 4 * KU) {
    for (int u = 0; u < KU; ++u) {
      Vec xv[MR];
      for (int i = 0; i < MR; ++i) xv[i] = vec_load(x + i * k + p + 4 * u);
      for (int j = 0; j < NR; ++j) {
        const Vec wv = vec_load(w + j * k + p + 4 * u);
        for (int i = 0; i < MR; ++i) acc[u][i][j] = vec_fma(acc[u][i][j], xv[i], wv);
      }
    }
  }
  // Fewer than 4 * KU left: whole vectors into set 0, then scalars.
  for (; p + 4 <= k; p += 4) {
    Vec xv[MR];
    for (int i = 0; i < MR; ++i) xv[i] = vec_load(x + i * k + p);
    for (int j = 0; j < NR; ++j) {
      const Vec wv = vec_load(w + j * k + p);
      for (int i = 0; i < MR; ++i) acc[0][i][j] = vec_fma(acc[0][i][j], xv[i], wv);
    }
  }

  for (int u = 1; u < KU; ++u)
    for (int i = 0; i < MR; ++i)
      for (int j = 0; j < NR; ++j) acc[0][i][j] = vec_add(acc[0][i][j], acc[u][i][j]);

  for (int i = 0; i < MR; ++i) {
    for (int j = 0; j < NR; ++j) {
      float s = vec_sum(acc[0][i][j]);
      for (std::int64_t q = p; q < k; ++q) s += x[i * k + q] * w[j * k + q];
      out[i * n + j] = s;
    }
  }
}

// One NR-row panel of W against every row of x, kRows rows at a time. The
// last m % kRows rows get a tile of exactly that height.
template <int NR>
void panel(const float* x, const float* w, float* out, std::int64_t m, std::int64_t k,
           std::int64_t n) {
  std::int64_t i = 0;
  for (; i + kRows <= m; i += kRows) tile<kRows, NR>(x + i * k, w, out + i * n, k, n);

  static_assert(kRows == 4, "the row tail below handles exactly kRows - 1 leftover heights");
  x += i * k;
  out += i * n;
  switch (m - i) {
    case 3: tile<3, NR>(x, w, out, k, n); break;
    case 2: tile<2, NR>(x, w, out, k, n); break;
    case 1: tile<1, NR>(x, w, out, k, n); break;
    default: break;
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

  // Threads split the output features into runs of whole panels: unit u is
  // columns [u * kCols, (u + 1) * kCols), the last unit holds the n % kCols
  // leftovers. Every column then goes through the same tile as on one
  // thread, so the result does not depend on the thread count.
  const auto run = [&](std::int64_t begin, std::int64_t end) {
    const std::int64_t j_end = std::min(end * kCols, n);
    std::int64_t j = begin * kCols;
    for (; j + kCols <= j_end; j += kCols) panel<kCols>(px, pw + j * k, po + j, m, k, n);
    // Leftover W rows one at a time; panel<1> still blocks over x's rows.
    for (; j < j_end; ++j) panel<1>(px, pw + j * k, po + j, m, k, n);
  };
  const std::int64_t unit_macs = std::max<std::int64_t>(1, m * k * kCols);
  kernel_parallel_for((n + kCols - 1) / kCols, (kMinChunkMacs + unit_macs - 1) / unit_macs, run);
}

}  // namespace minillm::ops
