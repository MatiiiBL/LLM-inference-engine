// Four float lanes, and the fp32 panel kernel that the Q8_0 and BF16 matmuls
// (q8_0.cpp, bf16.cpp) run for prefill.
//
// Private to src/ops/ — not part of the public interface.
//
// A weight format whose elements cost something to turn into floats cannot
// afford to convert them once per row of x: the cost would grow with m.
// Instead a panel of 4 * Vecs rows of W is converted once into a scratch
// buffer, exactly, so it holds the values of convert_to_f32(w), laid out with
// the 4 * Vecs values for one p together. A register tile of up to Rows rows
// of x by the panel's 4 * Vecs columns then runs every row block of x against
// the panel: each step broadcasts one element of x against the panel's Vecs
// vectors, so every output element is a single FMA chain over p ascending,
// the reference's order. Neither the tile's shape nor the panel's width
// changes any result. A 4 x 24 tile needs a load per 3.4 FMAs, and a 5 x 16
// one a load per 3.6, where ops::matmul's dot-product tile needs one per two,
// which is what pays for the conversion. Each format picks its shape
// (q8_0.cpp, bf16.cpp); both are measurements on an M4.
//
// On AArch64 the vectors are NEON. Elsewhere they are GCC/Clang vector
// extensions, which compute the same values up to FMA contraction.

#ifndef MINILLM_SRC_OPS_F32_PANEL_H
#define MINILLM_SRC_OPS_F32_PANEL_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace minillm::ops::detail {

// ---------------------------------------------------------------------------
// Four float lanes
// ---------------------------------------------------------------------------

#if defined(__aarch64__)

using Vec = float32x4_t;
inline Vec vec_zero() { return vdupq_n_f32(0.0f); }
inline Vec vec_load(const float* p) { return vld1q_f32(p); }
inline void vec_store(float* p, Vec v) { vst1q_f32(p, v); }
inline Vec vec_add(Vec a, Vec b) { return vaddq_f32(a, b); }
inline Vec vec_mul(Vec a, Vec b) { return vmulq_f32(a, b); }
inline Vec vec_scale(Vec a, float s) { return vmulq_n_f32(a, s); }
inline Vec vec_fma(Vec acc, Vec a, Vec b) { return vfmaq_f32(acc, a, b); }
inline Vec vec_fma_n(Vec acc, Vec a, float s) { return vfmaq_n_f32(acc, a, s); }
/// acc + a * b[L]
template <int L>
inline Vec vec_fma_lane(Vec acc, Vec a, Vec b) {
  return vfmaq_laneq_f32(acc, a, b, L);
}
inline float vec_sum(Vec v) { return vaddvq_f32(v); }

#else

typedef float Vec __attribute__((vector_size(16)));
inline Vec vec_splat(float s) { return Vec{s, s, s, s}; }
inline Vec vec_zero() { return vec_splat(0.0f); }
inline Vec vec_load(const float* p) {
  Vec v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}
inline void vec_store(float* p, Vec v) { std::memcpy(p, &v, sizeof(v)); }
inline Vec vec_add(Vec a, Vec b) { return a + b; }
inline Vec vec_mul(Vec a, Vec b) { return a * b; }
inline Vec vec_scale(Vec a, float s) { return a * vec_splat(s); }
inline Vec vec_fma(Vec acc, Vec a, Vec b) { return acc + a * b; }
inline Vec vec_fma_n(Vec acc, Vec a, float s) { return acc + a * vec_splat(s); }
template <int L>
inline Vec vec_fma_lane(Vec acc, Vec a, Vec b) {
  return acc + a * vec_splat(b[L]);
}
inline float vec_sum(Vec v) { return (v[0] + v[1]) + (v[2] + v[3]); }

#endif

// ---------------------------------------------------------------------------
// Panel kernel
// ---------------------------------------------------------------------------

/// One step of p for panel_tile: x[i, p + L] (lane L of xv[i]) times the
/// panel's row p + L, into every accumulator. Loading one panel vector at a
/// time keeps a 4 x 24 tile at 24 + 4 + 1 registers; loading a whole row
/// first spills.
template <int L, int Vecs, int MR>
inline void panel_step(Vec (&acc)[MR][Vecs], const float* row, const Vec (&xv)[MR]) {
  for (int v = 0; v < Vecs; ++v) {
    const Vec wv = vec_load(row + 4 * v);
    for (int i = 0; i < MR; ++i) acc[i][v] = vec_fma_lane<L>(acc[i][v], wv, xv[i]);
  }
}

/// out[i, c] for i < MR, c < 4 * Vecs: MR rows of x (row length k) against
/// the panel. out has row length ldo.
template <int Vecs, int MR>
void panel_tile(const float* x, std::int64_t k, const float* panel, float* out,
                std::int64_t ldo) {
  constexpr int kCols = 4 * Vecs;
  Vec acc[MR][Vecs];
  for (int i = 0; i < MR; ++i)
    for (int v = 0; v < Vecs; ++v) acc[i][v] = vec_zero();

  std::int64_t p = 0;
  for (; p + 4 <= k; p += 4) {
    Vec xv[MR];
    for (int i = 0; i < MR; ++i) xv[i] = vec_load(x + i * k + p);
    const float* row = panel + p * kCols;
    panel_step<0>(acc, row, xv);
    panel_step<1>(acc, row + kCols, xv);
    panel_step<2>(acc, row + 2 * kCols, xv);
    panel_step<3>(acc, row + 3 * kCols, xv);
  }
  // The last k % 4 values of p, one at a time, continuing the same chains.
  for (; p < k; ++p) {
    const float* row = panel + p * kCols;
    for (int v = 0; v < Vecs; ++v) {
      const Vec wv = vec_load(row + 4 * v);
      for (int i = 0; i < MR; ++i) acc[i][v] = vec_fma_n(acc[i][v], wv, x[i * k + p]);
    }
  }

  for (int i = 0; i < MR; ++i)
    for (int v = 0; v < Vecs; ++v) vec_store(out + i * ldo + 4 * v, acc[i][v]);
}

/// panel_tile<Vecs, rows> for rows in [1, MR].
template <int Vecs, int MR>
void panel_tile_rows(std::int64_t rows, const float* x, std::int64_t k, const float* panel,
                     float* out, std::int64_t ldo) {
  if constexpr (MR > 1) {
    if (rows < MR) {
      panel_tile_rows<Vecs, MR - 1>(rows, x, k, panel, out, ldo);
      return;
    }
  }
  panel_tile<Vecs, MR>(x, k, panel, out, ldo);
}

/// Panels [q_begin, q_end) of out = x · Wᵀ, x [m, k] and out [m, n], with a
/// tile of up to Rows rows of x by kCols = 4 * Vecs columns: panel q is output
/// columns [q * kCols, q * kCols + nc), nc = kCols except for the last panel
/// of a width that kCols does not divide. pack(j, nc, panel) must write rows
/// [j, j + nc) of W, converted exactly, into the k x kCols panel at
/// panel[p * kCols + c], and zero in columns nc and up. Every output element
/// is computed the same way whatever range of panels it falls in, which is
/// what makes a split over threads bitwise invariant.
template <int Vecs, int Rows, class Pack>
void panel_columns(const float* x, float* out, std::int64_t m, std::int64_t k, std::int64_t n,
                   std::int64_t q_begin, std::int64_t q_end, const Pack& pack) {
  constexpr int kCols = 4 * Vecs;
  // Rows of x in tiles of nearly equal height, Rows or one less once m is
  // large enough: a short last tile would hold too few accumulator chains to
  // hide the FMA latency. Heights do not change any result.
  const std::int64_t tiles = (m + Rows - 1) / Rows;
  const std::int64_t base = m / tiles;
  const std::int64_t taller = m % tiles;  // tiles with base + 1 rows

  // The panel: kCols * k floats, 55 KB for 24 columns at k = 576. One per
  // thread and tile shape, kept between calls.
  thread_local std::vector<float> panel;
  const std::size_t panel_floats = static_cast<std::size_t>(kCols * k);
  if (panel.size() < panel_floats) panel.resize(panel_floats);

  for (std::int64_t q = q_begin; q < q_end; ++q) {
    const std::int64_t j = q * kCols;
    const int nc = static_cast<int>(std::min<std::int64_t>(kCols, n - j));
    pack(j, nc, panel.data());
    std::int64_t i = 0;
    for (std::int64_t t = 0; t < tiles; ++t) {
      const std::int64_t rows = base + (t < taller ? 1 : 0);
      if (nc == kCols) {
        panel_tile_rows<Vecs, Rows>(rows, x + i * k, k, panel.data(), out + i * n + j, n);
      } else {
        // The last panel of a width that kCols does not divide: the tile
        // writes all its columns, so stage them and copy the real ones.
        float stage[Rows * kCols];
        panel_tile_rows<Vecs, Rows>(rows, x + i * k, k, panel.data(), stage, kCols);
        for (std::int64_t r = 0; r < rows; ++r) {
          std::memcpy(out + (i + r) * n + j, stage + r * kCols,
                      static_cast<std::size_t>(nc) * sizeof(float));
        }
      }
      i += rows;
    }
  }
}

}  // namespace minillm::ops::detail

#endif  // MINILLM_SRC_OPS_F32_PANEL_H
