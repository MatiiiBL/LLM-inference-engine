// A 4-lane float vector for the memory-bound kernels in norm.cpp and
// elementwise.cpp. Private to src/ops/ — not part of the public interface.
//
// Built on the GCC/Clang vector extension rather than intrinsics, so the same
// source lowers to NEON on arm64 and to SSE2 on x86-64 with no extra compiler
// flags, and every platform evaluates the same operations in the same order
// (only FMA contraction, which differs between compilers, moves last bits).
// Loads and stores go through memcpy: nothing here assumes alignment.

#ifndef MINILLM_SRC_OPS_F32X4_H
#define MINILLM_SRC_OPS_F32X4_H

#include <bit>
#include <cstdint>
#include <cstring>

namespace minillm::ops::detail {

typedef float f32x4 __attribute__((vector_size(16)));
typedef std::int32_t i32x4 __attribute__((vector_size(16)));  ///< lane masks: comparisons yield 0 / -1
typedef std::uint32_t u32x4 __attribute__((vector_size(16)));

inline constexpr std::int64_t kLanes = 4;

/// Floats per main-loop iteration in the reductions: four vectors, each with
/// its own accumulator, so four dependency chains run in parallel instead of
/// the reference's one.
inline constexpr std::int64_t kBlock = 4 * kLanes;

inline f32x4 splat(float v) { return f32x4{v, v, v, v}; }

inline f32x4 load(const float* p) {
  f32x4 v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

inline void store(float* p, f32x4 v) { std::memcpy(p, &v, sizeof(v)); }

/// Lane-wise max that never picks up a NaN from `v`: a lane only changes when
/// v > m, which is false for NaN. Same rule as the scalar `if (x > m) m = x`.
inline f32x4 max_keep(f32x4 m, f32x4 v) { return v > m ? v : m; }

/// (l0 + l1) + (l2 + l3) — a fixed order, so results do not depend on how the
/// compiler lowers the reduction.
inline float hsum(f32x4 v) { return (v[0] + v[1]) + (v[2] + v[3]); }

/// Largest lane, with the same NaN rule as max_keep.
inline float hmax(f32x4 v) {
  float m = v[0];
  for (int l = 1; l < 4; ++l) {
    if (v[l] > m) m = v[l];
  }
  return m;
}

/// Calls body(j) for every vector offset j = 0, 4, 8, ... with j + 4 <= n, in
/// increasing order, four calls per main-loop iteration. Clang does not unroll
/// loops written with explicit vectors, and at one vector per trip the loop
/// overhead, not memory, sets the speed. Returns where the scalar tail starts.
template <class Body>
inline std::int64_t for_each_vector(std::int64_t n, Body body) {
  std::int64_t j = 0;
  for (; j + kBlock <= n; j += kBlock) {
    body(j);
    body(j + kLanes);
    body(j + 2 * kLanes);
    body(j + 3 * kLanes);
  }
  for (; j + kLanes <= n; j += kLanes) body(j);
  return j;
}

/// True if any lane of a comparison mask is set.
inline bool any(i32x4 mask) {
  std::uint64_t halves[2];
  std::memcpy(halves, &mask, sizeof(halves));
  return (halves[0] | halves[1]) != 0;
}

/// max of p[0, n), n >= 1, with exactly the scalar rule `if (p[j] > m) m = p[j]`
/// starting from m = p[0]: a NaN is skipped unless it is p[0], in which case
/// it is the result. The maximum of a set does not depend on the order it is
/// visited in, so this returns the same value as that scalar loop (up to the
/// sign of a zero maximum).
inline float max_of(const float* p, std::int64_t n) {
  f32x4 m0 = splat(p[0]);
  f32x4 m1 = m0;
  f32x4 m2 = m0;
  f32x4 m3 = m0;
  std::int64_t j = 0;
  for (; j + kBlock <= n; j += kBlock) {
    m0 = max_keep(m0, load(p + j));
    m1 = max_keep(m1, load(p + j + kLanes));
    m2 = max_keep(m2, load(p + j + 2 * kLanes));
    m3 = max_keep(m3, load(p + j + 3 * kLanes));
  }
  for (; j + kLanes <= n; j += kLanes) m0 = max_keep(m0, load(p + j));
  float m = hmax(max_keep(max_keep(m0, m1), max_keep(m2, m3)));
  for (; j < n; ++j) {
    if (p[j] > m) m = p[j];
  }
  return m;
}

/// e^x lane-wise. Cephes expf: x = n ln2 + r with |r| <= ln2 / 2, then
/// e^r = 1 + r + r^2 P(r) with a degree-5 minimax P, then scale by 2^n
/// written straight into the exponent field.
///
/// Measured against double exp rounded to float, over every third float in
/// [-87.33, 88.37] (746M inputs): max error 1 ulp, mean 0.008 ulp, with and
/// without FMA contraction — the same worst case as the platform's expf.
/// Outside that range it trades the extremes for simplicity: x < -87.68
/// returns +0 where the true value is a subnormal below 1e-38, and x > 88.37
/// returns +inf where the true value may still be finite (it overflows at
/// 88.72). NaN propagates. Neither caller is sensitive to the extremes:
/// softmax only evaluates x <= 0, and silu_mul divides by 1 + e^x.
inline f32x4 exp(f32x4 x) {
  // Clamping keeps n in [-127, 128], where 2^n below is +0 .. +inf.
  const f32x4 lo = splat(-88.0f);
  const f32x4 hi = splat(89.0f);
  x = x < lo ? lo : x;
  x = x > hi ? hi : x;

  // n = round(x / ln2). Adding 1.5 * 2^23 pushes every fraction bit out of
  // the mantissa, so t's low bits hold n as a two's-complement integer — no
  // float-to-int conversion, so no undefined behaviour on NaN.
  const f32x4 magic = splat(12582912.0f);
  const f32x4 t = x * splat(1.44269504088896341f) + magic;
  const f32x4 n = t - magic;

  // r = x - n ln2, with ln2 split so that n * ln2_hi is exact.
  f32x4 r = x - n * splat(0.693359375f);
  r = r - n * splat(-2.12194440e-4f);

  f32x4 p = splat(1.9875691500e-4f);
  p = p * r + splat(1.3981999507e-3f);
  p = p * r + splat(8.3334519073e-3f);
  p = p * r + splat(4.1665795894e-2f);
  p = p * r + splat(1.6666665459e-1f);
  p = p * r + splat(5.0000001201e-1f);
  p = p * (r * r) + r + splat(1.0f);

  // Biased exponent n + 127 in [0, 255]: 0 is +0.0, 255 is +inf.
  const u32x4 biased = std::bit_cast<u32x4>(t) - std::bit_cast<u32x4>(magic) + u32x4{127, 127, 127, 127};
  return p * std::bit_cast<f32x4>(biased << 23);
}

}  // namespace minillm::ops::detail

#endif  // MINILLM_SRC_OPS_F32X4_H
