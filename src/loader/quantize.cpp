// quantize_q8_0 — owned by workflow 15-q8-quantize.
//
// loader.h specifies the quantizer exactly, so every implementation of it (the
// oracle's Python copy included) writes the same bytes. Per block of 32 floats
// x it rounds five times, each one IEEE-754 float operation:
//
//   1. amax / 127.0f          float division
//   2. f32_to_f16(...)        to the stored scale d, nearest even
//   3. 1.0f / f16_to_f32(d)   float reciprocal (f16_to_f32 itself is exact)
//   4. x * inv                float multiply
//   5. std::round(...)        to an integer, halves away from zero
//
// and then clamps to [-127, 127]. amax = max |x| is exact. The clamp can only
// bite when d is an f16 subnormal that rounded below amax / 127: a normal d
// keeps |x * inv| below 127.5.
//
// No step multiplies and then adds, so -ffp-contract=fast (GCC's default) has
// nothing to fuse into an FMA, and the bytes do not depend on the compiler.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "minillm/loader.h"

namespace minillm {
namespace {

[[noreturn]] void fail(const Tensor& src, const std::string& message) {
  throw std::invalid_argument("quantize_q8_0: " + src.to_string() + ": " + message);
}

/// The f16 bits of +inf: what f32_to_f16 returns for amax / 127 >= 65520.
constexpr std::uint16_t kF16Infinity = 0x7c00;

/// Quantizes the block of 32 floats at element `first` of `src`, which starts
/// at `x`.
void quantize_block(const Tensor& src, std::int64_t first, const float* x, BlockQ8_0& out) {
  // max |x|, taken on the bit patterns with the sign cleared: those order like
  // |x| for every non-NaN float, and every inf or NaN pattern is at least
  // kInfBits. One integer reduction finds amax and catches a non-finite value.
  constexpr std::uint32_t kInfBits = 0x7f800000u;
  std::uint32_t amax_bits = 0;
  for (std::int64_t j = 0; j < kQ8_0Block; ++j) {
    std::uint32_t bits;
    std::memcpy(&bits, &x[j], sizeof(bits));
    amax_bits = std::max(amax_bits, bits & 0x7fffffffu);
  }
  if (amax_bits >= kInfBits) {
    std::int64_t j = 0;
    while (std::isfinite(x[j])) ++j;
    fail(src, "element " + std::to_string(first + j) + " is " + std::to_string(x[j]) +
                  "; only finite values can be quantized");
  }
  float amax;
  std::memcpy(&amax, &amax_bits, sizeof(amax));

  out.d = f32_to_f16(amax / 127.0f);
  if (out.d == kF16Infinity) {
    fail(src, "the block of elements " + std::to_string(first) + ".." +
                  std::to_string(first + kQ8_0Block - 1) + " has amax " + std::to_string(amax) +
                  ", and amax / 127 overflows the f16 scale (65504 at most)");
  }
  const float d = f16_to_f32(out.d);
  if (d == 0.0f) {
    // amax / 127 rounded to 0 in f16 (an all-zero block, or one so small that
    // 1 / d would be inf): every value quantizes to 0.
    std::memset(out.qs, 0, sizeof(out.qs));
    return;
  }
  const float inv = 1.0f / d;
  for (std::int64_t j = 0; j < kQ8_0Block; ++j) {
    const float q = std::round(x[j] * inv);
    out.qs[j] = static_cast<std::int8_t>(std::clamp(q, -127.0f, 127.0f));
  }
}

}  // namespace

Tensor quantize_q8_0(const Tensor& src) {
  if (src.dtype() != DType::F32) fail(src, "only f32 tensors can be quantized");
  if (src.ndim() == 0) return Tensor();
  if (!src.is_contiguous()) fail(src, "the tensor must be contiguous");
  if (src.dim(src.ndim() - 1) % kQ8_0Block != 0) {
    fail(src, "the last dimension must be a multiple of " + std::to_string(kQ8_0Block));
  }

  Tensor out = Tensor::empty(src.shape(), DType::Q8_0);
  const float* x = src.f32();
  BlockQ8_0* blocks = out.data_as<BlockQ8_0>();
  const std::int64_t n_blocks = src.numel() / kQ8_0Block;
  for (std::int64_t b = 0; b < n_blocks; ++b) {
    quantize_block(src, b * kQ8_0Block, x + b * kQ8_0Block, blocks[b]);
  }
  return out;
}

}  // namespace minillm
