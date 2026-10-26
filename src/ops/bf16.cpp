// BF16 matmul and embedding — owned by workflow 20-bf16-kernels.
//
// A BF16 weight stands for exactly the F32 matrix convert_to_f32(w): a bf16
// is the top half of a float, so widening one is a 16-bit shift and exact.
// Activations stay F32 (ops.h).
//
// matmul has two kernels, chosen by the number of rows of x:
//
//   * m < kPanelMinRows, which includes decode (m = 1): ops::matmul's
//     dot-product tile, with each W vector widened as it is loaded, one
//     shift per four lanes (vshll_n_u16 on NEON). It streams W once, at half
//     the bytes of F32. A shift and an FMA per four weights fill all four
//     vector pipes of an M4 P-core at about 57 GB/s, so one thread is close
//     to the compute and the read ceiling at once (52 GB/s against a 64 GB/s
//     plain read), and twice as fast as F32. W rows go kDecodeRows at a time,
//     and for m > 1 x rows go up to four at a time so that a vector is
//     widened once for all of them.
//   * m >= kPanelMinRows (prefill): widening per row of x makes the cost
//     grow with m; the spike's per-row kernel was 1.74x slower than F32 at
//     m = 64. Instead a panel of kPanelCols rows of W is widened once into a
//     scratch buffer, transposed so that the values for one p are together,
//     and f32_panel.h's register tile runs every row block of x against it,
//     as the Q8_0 prefill kernel does, in a narrower shape.
//
// Both kernels are parallel over output columns (rows of W) and compute every
// output element the same way whatever the split, so the result is bitwise
// identical at every thread count.
//
// On AArch64 the kernels use NEON. Elsewhere the same code runs on GCC/Clang
// vector extensions, which is what CI's x86-64 GCC build exercises. It
// computes the same values, up to FMA contraction.

#include "bf16.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
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
// bf16 as floats
// ---------------------------------------------------------------------------

#if defined(__aarch64__)

/// w[0, 4) as floats, exactly.
inline Vec widen4(const std::uint16_t* w) {
  return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16));
}

/// w[0, 8) as floats, exactly: lo = w[0, 4), hi = w[4, 8).
inline void widen8(const std::uint16_t* w, Vec& lo, Vec& hi) {
  const uint16x8_t v = vld1q_u16(w);
  lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v), 16));
  hi = vreinterpretq_f32_u32(vshll_high_n_u16(v, 16));
}

#else

typedef std::uint16_t U16x4 __attribute__((vector_size(8)));
typedef std::uint16_t U16x8 __attribute__((vector_size(16)));
typedef std::uint32_t U32x4 __attribute__((vector_size(16)));

inline Vec widen4(const std::uint16_t* w) {
  U16x4 v;
  std::memcpy(&v, w, sizeof(v));
  return std::bit_cast<Vec>(__builtin_convertvector(v, U32x4) << 16);
}

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__

/// Interleaving with zeros puts each bf16 in the top half of a 32-bit lane:
/// one punpcklwd / punpckhwd per four lanes on SSE2, where widen4's zero
/// extension and shift take two.
inline void widen8(const std::uint16_t* w, Vec& lo, Vec& hi) {
  U16x8 v;
  std::memcpy(&v, w, sizeof(v));
  const U16x8 zero = {};
  lo = std::bit_cast<Vec>(__builtin_shufflevector(zero, v, 0, 8, 1, 9, 2, 10, 3, 11));
  hi = std::bit_cast<Vec>(__builtin_shufflevector(zero, v, 4, 12, 5, 13, 6, 14, 7, 15));
}

#else

inline void widen8(const std::uint16_t* w, Vec& lo, Vec& hi) {
  lo = widen4(w);
  hi = widen4(w + 4);
}

#endif

#endif

/// dst[0, len) = the floats src[0, len) stands for, bit for bit.
void widen_row(const std::uint16_t* src, float* dst, std::int64_t len) {
  std::int64_t p = 0;
  for (; p + 8 <= len; p += 8) {
    Vec lo;
    Vec hi;
    widen8(src + p, lo, hi);
    vec_store(dst + p, lo);
    vec_store(dst + p + 4, hi);
  }
  for (; p < len; ++p) dst[p] = bf16_to_f32(src[p]);
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

void require_bf16_contiguous_rank(const char* op, const char* name, const Tensor& t, int rank) {
  if (t.dtype() != DType::BF16) {
    fail(op, std::string(name) + " must be bf16, got " + t.to_string());
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
/// Measured on the model's projections: the panel kernel is ahead from m = 16
/// on at one thread and from m = 24 on at four. At ten, which calls this large
/// use by default, the decode kernel, whose units are 4 rows of W rather than
/// 16, is ahead up to m = 20 and level at m = 24-32.
constexpr std::int64_t kPanelMinRows = 24;

/// Rows of W per decode tile, and the unit threads split W into. Four, as in
/// ops::matmul. With two, a decode token measured 30% slower at one thread:
/// the m = 1 tile then has four FMA chains in flight, which caps it at about
/// one FMA per cycle, 34 GB/s of bf16. Four rows give eight chains, and with
/// a shift per FMA that is all four vector pipes.
constexpr int kDecodeRows = 4;

/// Smallest share of a call worth handing to another thread, in multiply-adds,
/// as in ops::matmul. A call with less than two of these runs inline.
constexpr std::int64_t kMinChunkMacs = std::int64_t{1} << 13;

/// out[i, r] for i < MR, r < NR: MR rows of x (row length k) starting at x,
/// against the NR rows of W (row length k) starting at w. out points at
/// element [0, 0] and has row length n.
///
/// Each element is a four-lane sum along k, in KU accumulator sets that take
/// alternate groups of four p, reduced once at the end, then the k % 4 tail
/// added in order.
template <int NR, int MR>
void decode_tile(const float* x, const std::uint16_t* w, float* out, std::int64_t k,
                 std::int64_t n) {
  // A tile with fewer than eight accumulators splits k across two sets, so
  // that enough FMA chains are in flight to hide their latency.
  constexpr int KU = NR * MR >= 8 ? 1 : 2;
  Vec acc[KU][NR][MR];
  for (int u = 0; u < KU; ++u)
    for (int r = 0; r < NR; ++r)
      for (int i = 0; i < MR; ++i) acc[u][r][i] = vec_zero();

  std::int64_t p = 0;
  for (; p + 8 <= k; p += 8) {
    Vec x0[MR];
    Vec x1[MR];
    for (int i = 0; i < MR; ++i) {
      x0[i] = vec_load(x + i * k + p);
      x1[i] = vec_load(x + i * k + p + 4);
    }
    for (int r = 0; r < NR; ++r) {
      Vec lo;
      Vec hi;
      widen8(w + r * k + p, lo, hi);
      for (int i = 0; i < MR; ++i) {
        acc[0][r][i] = vec_fma(acc[0][r][i], lo, x0[i]);
        acc[KU - 1][r][i] = vec_fma(acc[KU - 1][r][i], hi, x1[i]);
      }
    }
  }
  if (p + 4 <= k) {
    for (int r = 0; r < NR; ++r) {
      const Vec lo = widen4(w + r * k + p);
      for (int i = 0; i < MR; ++i) {
        acc[0][r][i] = vec_fma(acc[0][r][i], lo, vec_load(x + i * k + p));
      }
    }
    p += 4;
  }

  for (int r = 0; r < NR; ++r) {
    for (int i = 0; i < MR; ++i) {
      Vec v = acc[0][r][i];
      for (int u = 1; u < KU; ++u) v = vec_add(v, acc[u][r][i]);
      float s = vec_sum(v);
      for (std::int64_t q = p; q < k; ++q) s += x[i * k + q] * bf16_to_f32(w[r * k + q]);
      out[i * n + r] = s;
    }
  }
}

/// Rows [j, j + NR) of W against every row of x, x rows up to four at a time.
template <int NR>
void decode_rows(const float* x, const std::uint16_t* w, float* out, std::int64_t m,
                 std::int64_t k, std::int64_t n, std::int64_t j) {
  for (std::int64_t i = 0; i < m; i += 4) {
    const float* xi = x + i * k;
    const std::uint16_t* wj = w + j * k;
    float* o = out + i * n + j;
    switch (std::min<std::int64_t>(4, m - i)) {
      case 1: decode_tile<NR, 1>(xi, wj, o, k, n); break;
      case 2: decode_tile<NR, 2>(xi, wj, o, k, n); break;
      case 3: decode_tile<NR, 3>(xi, wj, o, k, n); break;
      default: decode_tile<NR, 4>(xi, wj, o, k, n); break;
    }
  }
}

/// Rows [j_begin, j_end) of W: NR at a time, the leftovers one at a time.
template <int NR>
void decode_columns(const float* x, const std::uint16_t* w, float* out, std::int64_t m,
                    std::int64_t k, std::int64_t n, std::int64_t j_begin, std::int64_t j_end) {
  std::int64_t j = j_begin;
  for (; j + NR <= j_end; j += NR) decode_rows<NR>(x, w, out, m, k, n, j);
  for (; j < j_end; ++j) decode_rows<1>(x, w, out, m, k, n, j);
}

void matmul_decode(const float* x, const std::uint16_t* w, float* out, std::int64_t m,
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

/// The tile: up to kPanelRows rows of x by a panel of kPanelCols rows of W,
/// 20 accumulators. Q8_0's 4 x 24 tile measured the same at one and four
/// threads, but at ten (the default, which large calls use) the 4 + 6 cores
/// of an M4 leave too few 24-row units on k/v_proj (8), q/o_proj and
/// down_proj (24) to share out: prefill at m = 64 was 0.98x F32 there, and
/// is 1.09x with 16-row panels. 6 x 12 measured 1-2% slower at one thread.
constexpr int kPanelVecs = 4;               ///< vectors of four columns per tile row
constexpr int kPanelCols = 4 * kPanelVecs;  ///< rows of W per panel: 16 divides 192, 576, 1536
constexpr int kPanelRows = 5;               ///< rows of x per tile

/// Column c < nc of the panel: row c of W (at w, k values), widened into
/// panel[p * kPanelCols + c] for every p.
void pack_column(const std::uint16_t* w, std::int64_t k, float* panel) {
  for (std::int64_t p = 0; p < k; ++p) panel[p * kPanelCols] = bf16_to_f32(w[p]);
}

/// Four adjacent columns of the panel at once: rows w, w + k, w + 2 k and
/// w + 3 k of W into panel[p * kPanelCols + c] for c < 4. The same values as
/// pack_column, which is what the portable build uses.
void pack_4_columns(const std::uint16_t* w, std::int64_t k, float* panel) {
#if defined(__aarch64__)
  const std::uint16_t* rows[4] = {w, w + k, w + 2 * k, w + 3 * k};
  std::int64_t p = 0;
  for (; p + 8 <= k; p += 8) {
    // Transpose 4 x 8 bf16 so that each 64-bit half holds columns 0..3 of
    // one p: zip the rows pairwise, then zip the pairs. Widening comes last,
    // on half as many bytes.
    const uint16x8_t r0 = vld1q_u16(rows[0] + p);
    const uint16x8_t r1 = vld1q_u16(rows[1] + p);
    const uint16x8_t r2 = vld1q_u16(rows[2] + p);
    const uint16x8_t r3 = vld1q_u16(rows[3] + p);
    const uint32x4_t a0 = vreinterpretq_u32_u16(vzip1q_u16(r0, r1));
    const uint32x4_t a1 = vreinterpretq_u32_u16(vzip2q_u16(r0, r1));
    const uint32x4_t b0 = vreinterpretq_u32_u16(vzip1q_u16(r2, r3));
    const uint32x4_t b1 = vreinterpretq_u32_u16(vzip2q_u16(r2, r3));
    const uint16x8_t groups[4] = {
        vreinterpretq_u16_u32(vzip1q_u32(a0, b0)), vreinterpretq_u16_u32(vzip2q_u32(a0, b0)),
        vreinterpretq_u16_u32(vzip1q_u32(a1, b1)), vreinterpretq_u16_u32(vzip2q_u32(a1, b1))};
    float* dst = panel + p * kPanelCols;
    for (int g = 0; g < 4; ++g) {
      vec_store(dst + (2 * g) * kPanelCols,
                vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(groups[g]), 16)));
      vec_store(dst + (2 * g + 1) * kPanelCols,
                vreinterpretq_f32_u32(vshll_high_n_u16(groups[g], 16)));
    }
  }
  for (; p < k; ++p) {
    for (int c = 0; c < 4; ++c) panel[p * kPanelCols + c] = bf16_to_f32(rows[c][p]);
  }
#else
  for (int c = 0; c < 4; ++c) pack_column(w + c * k, k, panel + c);
#endif
}

/// Rows [0, nc) of W at w (k values each), nc <= kPanelCols, widened into the
/// k x kPanelCols panel; columns nc and up are zero.
void pack_panel(const std::uint16_t* w, std::int64_t k, int nc, float* panel) {
  int c = 0;
  for (; c + 4 <= nc; c += 4) pack_4_columns(w + c * k, k, panel + c);
  for (; c < nc; ++c) pack_column(w + c * k, k, panel + c);
  if (nc < kPanelCols) {
    for (std::int64_t p = 0; p < k; ++p) {
      std::fill(panel + p * kPanelCols + nc, panel + (p + 1) * kPanelCols, 0.0f);
    }
  }
}

void matmul_panel(const float* x, const std::uint16_t* w, float* out, std::int64_t m,
                  std::int64_t k, std::int64_t n) {
  const auto pack = [&](std::int64_t j, int nc, float* panel) {
    pack_panel(w + j * k, k, nc, panel);
  };
  const auto run = [&](std::int64_t begin, std::int64_t end) {
    panel_columns<kPanelVecs, kPanelRows>(x, out, m, k, n, begin, end, pack);
  };
  const std::int64_t unit_macs = std::max<std::int64_t>(1, m * k * kPanelCols);
  kernel_parallel_for((n + kPanelCols - 1) / kPanelCols,
                      (kMinChunkMacs + unit_macs - 1) / unit_macs, m * k * n, std::ref(run));
}

}  // namespace

void matmul_bf16(const Tensor& x, const Tensor& w, Tensor& out) {
  require_f32_contiguous_rank("matmul", "x", x, 2);
  require_bf16_contiguous_rank("matmul", "w", w, 2);
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

  const float* px = x.f32();
  const std::uint16_t* pw = w.data_as<std::uint16_t>();
  if (m < kPanelMinRows) {
    matmul_decode(px, pw, po, m, k, n);
  } else {
    matmul_panel(px, pw, po, m, k, n);
  }
}

void embedding_bf16(const Tensor& table, std::span<const std::int32_t> ids, Tensor& out) {
  require_bf16_contiguous_rank("embedding", "table", table, 2);
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

  const std::uint16_t* src = table.data_as<std::uint16_t>();
  float* dst = out.f32();
  for (std::int64_t t = 0; t < n_tokens; ++t) {
    const std::int64_t id = ids[static_cast<std::size_t>(t)];
    widen_row(src + id * hidden, dst + t * hidden, hidden);
  }
}

}  // namespace minillm::ops::detail
