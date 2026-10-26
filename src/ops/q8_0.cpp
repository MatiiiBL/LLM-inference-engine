// Q8_0 matmul and embedding — owned by workflow 16-q8-kernels.
//
// A Q8_0 weight stands for exactly the F32 matrix convert_to_f32(w): element
// j of a block is float(qs[j]) * d, which is exact in float (an integer of at
// most 7 bits times an 11-bit significand). Activations stay F32 (ops.h).
//
// matmul has two kernels, chosen by the number of rows of x:
//
//   * m < kPanelMinRows, which includes decode (m = 1): a matrix-vector style
//     product that streams W once. Per block it widens the 32 int8 to float,
//     takes their dot product with x unscaled, and folds the scale in with one
//     FMA, d * sum(x * q): a reassociation of the reference's sum(x * (q * d)),
//     which ops.h allows. The int8-to-float conversion, not memory, sets the
//     speed: about 30 vector ops per block of 32 weights, where fp32 needs 8
//     loads and 8 FMAs but moves 3.76 times the bytes. W rows go
//     kDecodeRows at a time, and for m > 1 x rows go up to four at a time so
//     that a block is converted once for all of them.
//   * m >= kPanelMinRows (prefill): converting per row of x makes the cost
//     grow with m; the spike's per-row kernel was 3.3x slower than fp32 at
//     m = 64. Instead a panel of kPanelCols rows of W is dequantized once into
//     a scratch buffer, exactly, so it holds the values of convert_to_f32(w),
//     laid out with the kPanelCols values for one p together. A kPanelRows x
//     kPanelCols register tile then runs every row block of x against the
//     panel: each step broadcasts one element of x against kPanelCols / 4
//     vectors of the panel, so every output element is a single FMA chain over
//     p ascending, the reference's order. The tile needs a load per 3.4 FMAs
//     where ops::matmul's dot-product tile needs one per two, which is what
//     pays for the dequantization. kPanelMinRows, the tile and the panel width
//     are measurements on an M4 P-core, in cycles.
//
// Both kernels are parallel over output columns (rows of W) and compute every
// output element the same way whatever the split, so the result is bitwise
// identical at every thread count.
//
// On AArch64 the kernels use NEON. Elsewhere the same code runs on GCC/Clang
// vector extensions and a scalar int8 widening; it computes the same values,
// up to FMA contraction, and is what CI's x86-64 GCC build exercises. It is a
// correctness fallback: on that runner Q8_0 decode is about half fp32's speed,
// because SSE2 has no cheap int8-to-float widening.

#include "q8_0.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "f32_panel.h"
#include "minillm/runtime.h"
#include "validate.h"

namespace minillm::ops::detail {
namespace {

// ---------------------------------------------------------------------------
// A block's int8 as floats (Vec and its operations are in f32_panel.h)
// ---------------------------------------------------------------------------

#if defined(__aarch64__)

/// A block's scale as float: the hardware conversion, exact like tensor.h's
/// f16_to_f32 (which the portable build uses).
inline float scale_of(const BlockQ8_0& block) {
  return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(block.d))), 0);
}

/// TBL indices that move bytes 4i .. 4i + 3 of a register into the top byte
/// of each 32-bit lane and zero the rest (index 255 is out of range).
alignas(16) constexpr std::uint8_t kTopByte[4][16] = {
    {255, 255, 255, 0, 255, 255, 255, 1, 255, 255, 255, 2, 255, 255, 255, 3},
    {255, 255, 255, 4, 255, 255, 255, 5, 255, 255, 255, 6, 255, 255, 255, 7},
    {255, 255, 255, 8, 255, 255, 255, 9, 255, 255, 255, 10, 255, 255, 255, 11},
    {255, 255, 255, 12, 255, 255, 255, 13, 255, 255, 255, 14, 255, 255, 255, 15}};

/// 16 int8 (spread four per 32-bit lane group by `idx`) to float, exactly:
/// TBL leaves q * 2^24 in each lane and a fixed-point SCVTF with 24 fraction
/// bits gives float(q). Two ops per four lanes. The spike's widening through
/// int16 and fp16 takes as many ops but measured about 15% slower per token.
inline Vec widen4(int8x16_t q, uint8x16_t idx) {
  return vcvtq_n_f32_s32(vreinterpretq_s32_s8(vqtbl1q_s8(q, idx)), 24);
}

/// f[l] = float(qs[4l .. 4l + 3]).
inline void widen_block(const std::int8_t* qs, Vec f[8]) {
  const int8x16_t lo = vld1q_s8(qs);
  const int8x16_t hi = vld1q_s8(qs + 16);
  for (int i = 0; i < 4; ++i) {
    const uint8x16_t idx = vld1q_u8(kTopByte[i]);
    f[i] = widen4(lo, idx);
    f[4 + i] = widen4(hi, idx);
  }
}

#else

inline float scale_of(const BlockQ8_0& block) { return f16_to_f32(block.d); }

/// Plain scalar conversions. CI's GCC 13 on x86-64 ran the prefill kernel
/// 20% slower with __builtin_convertvector here, and decode no faster.
inline void widen_block(const std::int8_t* qs, Vec f[8]) {
  for (int l = 0; l < 8; ++l) {
    const std::int8_t* q = qs + 4 * l;
    f[l] = Vec{static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]),
               static_cast<float>(q[3])};
  }
}

#endif

/// out[0, 32) = convert_to_f32 of one block, bit for bit: float(q) * d is
/// exact, so the order of anything here cannot matter.
inline void dequantize_block(const BlockQ8_0& block, float* out) {
  Vec f[8];
  widen_block(block.qs, f);
  const float d = scale_of(block);
  for (int l = 0; l < 8; ++l) vec_store(out + 4 * l, vec_scale(f[l], d));
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

void require_q8_0_contiguous_rank(const char* op, const char* name, const Tensor& t, int rank) {
  if (t.dtype() != DType::Q8_0) {
    fail(op, std::string(name) + " must be q8_0, got " + t.to_string());
  }
  if (!t.is_contiguous()) {
    fail(op, std::string(name) + " must be contiguous, got " + t.to_string());
  }
  if (t.ndim() != rank) {
    fail(op, std::string(name) + " must have rank " + std::to_string(rank) + ", got " +
                 t.to_string());
  }
}

// ---------------------------------------------------------------------------
// Decode kernel
// ---------------------------------------------------------------------------

/// Below this many rows of x, the decode kernel; from it on, the panel kernel.
/// Measured in cycles at every model shape: the decode kernel wins up to
/// m = 3-5 depending on the shape, the panel kernel from m = 6-8 on.
constexpr std::int64_t kPanelMinRows = 6;

/// Rows of W per decode tile, and the unit threads split W into. Two, not
/// ops::matmul's four: at one thread it makes no difference (the kernel is
/// compute-bound at 20 GB/s either way), and at four threads it streams
/// 7-12% faster on the projections. A Q8_0 row of 576 is only five cache
/// lines, and two interleaved streams keep the prefetcher ahead better than
/// four short ones.
constexpr int kDecodeRows = 2;

/// Smallest share of a call worth handing to another thread, in multiply-adds,
/// as in ops::matmul. A call with less than two of these runs inline.
constexpr std::int64_t kMinChunkMacs = std::int64_t{1} << 13;

/// out[i, j] for i < MR, j < NR: MR rows of x (row length k) starting at x,
/// against the NR rows of W (nb blocks each) starting at w. out points at
/// element [0, 0] and has row length n.
///
/// Every element is computed the same way whatever NR and MR are: per block,
/// s = q[0..3]·x[0..3] then seven FMAs in block order, all four lanes apart;
/// then acc = fma(acc, s, d); then the four lanes summed once at the end.
template <int NR, int MR>
void decode_tile(const float* x, const BlockQ8_0* w, float* out, std::int64_t k,
                 std::int64_t nb, std::int64_t n) {
  Vec acc[NR][MR];
  for (int r = 0; r < NR; ++r)
    for (int i = 0; i < MR; ++i) acc[r][i] = vec_zero();

  for (std::int64_t b = 0; b < nb; ++b) {
    for (int r = 0; r < NR; ++r) {
      const BlockQ8_0& block = w[r * nb + b];
      Vec f[8];
      widen_block(block.qs, f);
      const float d = scale_of(block);
      for (int i = 0; i < MR; ++i) {
        const float* xb = x + i * k + b * kQ8_0Block;
        Vec s = vec_mul(f[0], vec_load(xb));
        for (int l = 1; l < 8; ++l) s = vec_fma(s, f[l], vec_load(xb + 4 * l));
        acc[r][i] = vec_fma_n(acc[r][i], s, d);
      }
    }
  }

  for (int r = 0; r < NR; ++r)
    for (int i = 0; i < MR; ++i) out[i * n + r] = vec_sum(acc[r][i]);
}

/// Rows [j, j + NR) of W against every row of x, x rows up to four at a time.
template <int NR>
void decode_rows(const float* x, const BlockQ8_0* w, float* out, std::int64_t m,
                 std::int64_t k, std::int64_t n, std::int64_t j) {
  const std::int64_t nb = k / kQ8_0Block;
  for (std::int64_t i = 0; i < m; i += 4) {
    const float* xi = x + i * k;
    const BlockQ8_0* wj = w + j * nb;
    float* o = out + i * n + j;
    switch (std::min<std::int64_t>(4, m - i)) {
      case 1: decode_tile<NR, 1>(xi, wj, o, k, nb, n); break;
      case 2: decode_tile<NR, 2>(xi, wj, o, k, nb, n); break;
      case 3: decode_tile<NR, 3>(xi, wj, o, k, nb, n); break;
      default: decode_tile<NR, 4>(xi, wj, o, k, nb, n); break;
    }
  }
}

/// Rows [j_begin, j_end) of W: NR at a time, the leftovers one at a time.
template <int NR>
void decode_columns(const float* x, const BlockQ8_0* w, float* out, std::int64_t m,
                    std::int64_t k, std::int64_t n, std::int64_t j_begin, std::int64_t j_end) {
  std::int64_t j = j_begin;
  for (; j + NR <= j_end; j += NR) decode_rows<NR>(x, w, out, m, k, n, j);
  for (; j < j_end; ++j) decode_rows<1>(x, w, out, m, k, n, j);
}

void matmul_decode(const float* x, const BlockQ8_0* w, float* out, std::int64_t m,
                   std::int64_t k, std::int64_t n) {
  const auto run = [&](std::int64_t begin, std::int64_t end) {
    decode_columns<kDecodeRows>(x, w, out, m, k, n, begin * kDecodeRows,
                                std::min(end * kDecodeRows, n));
  };
  const std::int64_t unit_macs = std::max<std::int64_t>(1, m * k * kDecodeRows);
  kernel_parallel_for((n + kDecodeRows - 1) / kDecodeRows,
                      (kMinChunkMacs + unit_macs - 1) / unit_macs, m * k * n, std::ref(run));
}

// ---------------------------------------------------------------------------
// Panel kernel (the tile is in f32_panel.h)
// ---------------------------------------------------------------------------

constexpr int kPanelVecs = 6;               ///< vectors of four columns per tile row
constexpr int kPanelCols = 4 * kPanelVecs;  ///< rows of W per panel: 24 divides 192, 576, 1536
constexpr int kPanelRows = 4;               ///< rows of x per tile: 24 accumulators

/// Column c < nc of the panel: row c of W (at w, nb blocks), dequantized into
/// panel[p * kPanelCols + c] for every p.
void pack_column(const BlockQ8_0* w, std::int64_t nb, float* panel) {
  float values[kQ8_0Block];
  for (std::int64_t b = 0; b < nb; ++b) {
    dequantize_block(w[b], values);
    float* dst = panel + b * kQ8_0Block * kPanelCols;
    for (std::int64_t p = 0; p < kQ8_0Block; ++p) dst[p * kPanelCols] = values[p];
  }
}

/// Four adjacent columns of the panel at once: rows w, w + nb, w + 2 nb and
/// w + 3 nb of W into panel[p * kPanelCols + c] for c < 4. The same values as
/// pack_column, which is what the portable build uses.
void pack_4_columns(const BlockQ8_0* w, std::int64_t nb, float* panel) {
#if defined(__aarch64__)
  const BlockQ8_0* rows[4] = {w, w + nb, w + 2 * nb, w + 3 * nb};
  uint8x16_t idx[4];
  for (int i = 0; i < 4; ++i) idx[i] = vld1q_u8(kTopByte[i]);
  for (std::int64_t b = 0; b < nb; ++b) {
    uint16x4_t bits = vdup_n_u16(0);
    bits = vld1_lane_u16(&rows[0][b].d, bits, 0);
    bits = vld1_lane_u16(&rows[1][b].d, bits, 1);
    bits = vld1_lane_u16(&rows[2][b].d, bits, 2);
    bits = vld1_lane_u16(&rows[3][b].d, bits, 3);
    const Vec d = vcvt_f32_f16(vreinterpret_f16_u16(bits));  // one scale per column
    for (int half = 0; half < 2; ++half) {
      // Interleave the four rows' bytes so that each 32-bit group holds
      // column 0..3 of one p: zip bytes pairwise, then zip the pairs.
      const int8x16_t r0 = vld1q_s8(rows[0][b].qs + 16 * half);
      const int8x16_t r1 = vld1q_s8(rows[1][b].qs + 16 * half);
      const int8x16_t r2 = vld1q_s8(rows[2][b].qs + 16 * half);
      const int8x16_t r3 = vld1q_s8(rows[3][b].qs + 16 * half);
      const int16x8_t a0 = vreinterpretq_s16_s8(vzip1q_s8(r0, r1));
      const int16x8_t a1 = vreinterpretq_s16_s8(vzip2q_s8(r0, r1));
      const int16x8_t b0 = vreinterpretq_s16_s8(vzip1q_s8(r2, r3));
      const int16x8_t b1 = vreinterpretq_s16_s8(vzip2q_s8(r2, r3));
      const int8x16_t groups[4] = {
          vreinterpretq_s8_s16(vzip1q_s16(a0, b0)), vreinterpretq_s8_s16(vzip2q_s16(a0, b0)),
          vreinterpretq_s8_s16(vzip1q_s16(a1, b1)), vreinterpretq_s8_s16(vzip2q_s16(a1, b1))};
      float* dst = panel + (b * kQ8_0Block + 16 * half) * kPanelCols;
      for (int g = 0; g < 4; ++g) {
        for (int l = 0; l < 4; ++l) {
          vec_store(dst + (4 * g + l) * kPanelCols, vec_mul(widen4(groups[g], idx[l]), d));
        }
      }
    }
  }
#else
  for (int c = 0; c < 4; ++c) pack_column(w + c * nb, nb, panel + c);
#endif
}

/// Rows [0, nc) of W at w (nb blocks each), nc <= kPanelCols, dequantized into
/// the k x kPanelCols panel; columns nc and up are zero.
void pack_panel(const BlockQ8_0* w, std::int64_t nb, int nc, float* panel) {
  int c = 0;
  for (; c + 4 <= nc; c += 4) pack_4_columns(w + c * nb, nb, panel + c);
  for (; c < nc; ++c) pack_column(w + c * nb, nb, panel + c);
  if (nc < kPanelCols) {
    const std::int64_t k = nb * kQ8_0Block;
    for (std::int64_t p = 0; p < k; ++p) {
      std::fill(panel + p * kPanelCols + nc, panel + (p + 1) * kPanelCols, 0.0f);
    }
  }
}

void matmul_panel(const float* x, const BlockQ8_0* w, float* out, std::int64_t m,
                  std::int64_t k, std::int64_t n) {
  const std::int64_t nb = k / kQ8_0Block;
  const auto pack = [&](std::int64_t j, int nc, float* panel) {
    pack_panel(w + j * nb, nb, nc, panel);
  };
  const auto run = [&](std::int64_t begin, std::int64_t end) {
    panel_columns<kPanelVecs, kPanelRows>(x, out, m, k, n, begin, end, pack);
  };
  const std::int64_t unit_macs = std::max<std::int64_t>(1, m * k * kPanelCols);
  kernel_parallel_for((n + kPanelCols - 1) / kPanelCols,
                      (kMinChunkMacs + unit_macs - 1) / unit_macs, m * k * n, std::ref(run));
}

}  // namespace

void matmul_q8_0(const Tensor& x, const Tensor& w, Tensor& out) {
  require_f32_contiguous_rank("matmul", "x", x, 2);
  require_q8_0_contiguous_rank("matmul", "w", w, 2);
  require_f32_contiguous_rank("matmul", "out", out, 2);

  const std::int64_t m = x.shape()[0];
  const std::int64_t k = x.shape()[1];
  const std::int64_t n = w.shape()[0];

  require_dim("matmul", "w", w, 1, k, "in_features must match x width");
  require_dim("matmul", "out", out, 0, m, "rows of x");
  require_dim("matmul", "out", out, 1, n, "out_features of w");

  if (m == 0 || n == 0) return;
  float* po = out.f32();
  if (k == 0) {  // an empty sum; x and w may be wrap()s of nullptr
    std::fill_n(po, m * n, 0.0f);
    return;
  }

  // Q8_0's shape rule makes k a multiple of the block, which both kernels
  // rely on (the panel tile also needs k % 4 == 0).
  const float* px = x.f32();
  const BlockQ8_0* pw = w.data_as<BlockQ8_0>();
  if (m < kPanelMinRows) {
    matmul_decode(px, pw, po, m, k, n);
  } else {
    matmul_panel(px, pw, po, m, k, n);
  }
}

void embedding_q8_0(const Tensor& table, std::span<const std::int32_t> ids, Tensor& out) {
  require_q8_0_contiguous_rank("embedding", "table", table, 2);
  require_f32_contiguous_rank("embedding", "out", out, 2);

  const std::int64_t n_tokens = static_cast<std::int64_t>(ids.size());
  const std::int64_t vocab_size = table.shape()[0];
  const std::int64_t hidden = table.shape()[1];

  require_dim("embedding", "out", out, 0, n_tokens, "one row per id");
  require_dim("embedding", "out", out, 1, hidden, "table width");

  // Every id is checked before anything is written, so a bad one leaves out
  // untouched.
  for (const std::int32_t id : ids) {
    if (id < 0 || id >= vocab_size) {
      throw std::out_of_range("ops::embedding: token id " + std::to_string(id) +
                              " out of range for vocab size " + std::to_string(vocab_size));
    }
  }

  // A zero-width table may be a wrap() of nullptr.
  if (hidden == 0) return;

  const std::int64_t nb = hidden / kQ8_0Block;
  const BlockQ8_0* src = table.data_as<BlockQ8_0>();
  float* dst = out.f32();
  for (std::int64_t t = 0; t < n_tokens; ++t) {
    const BlockQ8_0* row = src + static_cast<std::int64_t>(ids[static_cast<std::size_t>(t)]) * nb;
    for (std::int64_t b = 0; b < nb; ++b) dequantize_block(row[b], dst + t * hidden + b * kQ8_0Block);
  }
}

}  // namespace minillm::ops::detail
