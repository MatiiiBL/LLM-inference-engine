// Q8_0 design spike — workflow 14. NOT part of the engine or of the CMake
// build. It exists so that every number in docs/proposals/14-q8_0.md can be
// reproduced, and it changes nothing under include/ or src/: it links the
// engine's own tensor, ops and attention sources as they are on main.
//
// Build, from the repo root (one command):
//
//   mkdir -p build && clang++ -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Iinclude -Itests \
//     -o build/q8_0_spike spike/q8_0/q8_0_spike.cpp src/tensor/tensor.cpp \
//     src/ops/matmul.cpp src/ops/norm.cpp src/ops/elementwise.cpp src/ops/rope.cpp \
//     src/attention/attention.cpp src/ops_ref/matmul.cpp src/ops_ref/attention.cpp \
//     tests/common/npy.cpp
//
// -O3 -DNDEBUG is exactly what the release preset passes; no -mcpu, so the
// target is AppleClang's default apple-m1, which already has dotprod and
// fp16 vector arithmetic (any Apple Silicon Mac).
//
// Run:
//
//   build/q8_0_spike <model_dir> <oracle_dir> [selftest] [bench] [threads] [quality | refcheck]
//                    [torchq8=<dir>]
//
// With no section names it runs the first four. refcheck is the last part of
// quality alone: the Q8_0 kernels against their references, and against the
// output of torch_q8_oracle.py when torchq8=<dir> names it. Run it from the
// repo root (it reads spike/q8_0/eval_ids.txt); model_dir is read only.
//
// Format under test (llama.cpp's Q8_0): blocks of 32 consecutive weights along
// a row, each stored as 32 int8 plus one scale d, value = q * d. Two block
// layouts are measured: BlockQ8_0 with an fp16 scale (34 bytes, llama.cpp's
// block_q8_0) and BlockQ8_0F with an fp32 scale (36 bytes).
//
// Two kernels, both for m = 1 (decode), 4 weight rows at a time like
// ops::matmul (its comment explains why 4 streams):
//   dequant  int8 -> fp32 on the fly, fp32 FMAs against fp32 activations.
//   w8a8     the activation row is quantized to Q8_0 too (fp32 scale), and
//            each block pair is one vdotq_s32 pair, exact in int32, scaled by
//            d_w * d_x.

#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/npy.h"
#include "common/tolerances.h"
#include "minillm/ops.h"
#include "minillm/tensor.h"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_DOTPROD) || \
    !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error "the spike's kernels need NEON with dotprod and fp16 (any Apple Silicon Mac, default flags)"
#endif

namespace fs = std::filesystem;
using namespace minillm;

namespace {

// ===========================================================================
// Small utilities
// ===========================================================================

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::string load_string() {
  double l[3] = {0, 0, 0};
  if (getloadavg(l, 3) != 3) return "load average unavailable";
  char buf[96];
  std::snprintf(buf, sizeof(buf), "load average %.2f %.2f %.2f", l[0], l[1], l[2]);
  return buf;
}

// f32 -> f16 bits, round to nearest even (Giesen's float_to_half_fast3_rtne).
// tensor.h has f16_to_f32 but not this direction; the proposal adds it next to
// f32_to_bf16. selftest checks it against the hardware fcvt on every
// non-negative float.
std::uint16_t f32_to_f16(float f) {
  constexpr std::uint32_t kF32Infinity = 255u << 23;
  constexpr std::uint32_t kF16Max = (127u + 16u) << 23;
  constexpr std::uint32_t kDenormMagic = ((127u - 15u) + (23u - 10u) + 1u) << 23;
  std::uint32_t u = std::bit_cast<std::uint32_t>(f);
  const std::uint32_t sign = u & 0x80000000u;
  u ^= sign;
  std::uint32_t o;
  if (u >= kF16Max) {
    o = u > kF32Infinity ? 0x7e00u : 0x7c00u;  // NaN stays NaN, the rest overflows
  } else if (u < (113u << 23)) {
    // f16 subnormal or zero: let the fp adder do the rounding.
    const float r = std::bit_cast<float>(u) + std::bit_cast<float>(kDenormMagic);
    o = std::bit_cast<std::uint32_t>(r) - kDenormMagic;
  } else {
    const std::uint32_t mant_odd = (u >> 13) & 1u;
    u += static_cast<std::uint32_t>(15 - 127) << 23;  // rebias (wraps, on purpose)
    u += 0xfffu + mant_odd;
    o = u >> 13;
  }
  return static_cast<std::uint16_t>(o | (sign >> 16));
}

// Hardware f16 -> f32 (one fcvt). tensor.h's f16_to_f32 is the portable,
// branchy version; the kernels use this one.
inline float f16_hw(std::uint16_t h) { return static_cast<float>(std::bit_cast<_Float16>(h)); }

// ===========================================================================
// Minimal JSON, enough for a safetensors header and config.json. The engine
// uses nlohmann/json, which the spike cannot reach without CMake.
// ===========================================================================

struct Json {
  enum class Kind { kNull, kBool, kNumber, kString, kArray, kObject } kind = Kind::kNull;
  double number = 0;
  bool boolean = false;
  std::string str;
  std::vector<Json> items;
  std::vector<std::pair<std::string, Json>> members;

  const Json* get(std::string_view key) const {
    for (const auto& [k, v] : members)
      if (k == key) return &v;
    return nullptr;
  }
  const Json& at(std::string_view key) const {
    const Json* v = get(key);
    if (v == nullptr) throw std::runtime_error("json: missing key " + std::string(key));
    return *v;
  }
};

class JsonParser {
 public:
  explicit JsonParser(std::string_view s) : s_(s) {}
  Json parse() {
    Json v = value();
    ws();
    if (i_ != s_.size()) fail("trailing characters");
    return v;
  }

 private:
  [[noreturn]] void fail(const char* what) {
    throw std::runtime_error(std::string("json: ") + what + " at offset " + std::to_string(i_));
  }
  void ws() {
    while (i_ < s_.size() && std::strchr(" \t\r\n", s_[i_]) != nullptr) ++i_;
  }
  char peek() {
    ws();
    if (i_ >= s_.size()) fail("unexpected end");
    return s_[i_];
  }
  void expect(char c) {
    if (peek() != c) fail("unexpected character");
    ++i_;
  }
  std::string string() {
    expect('"');
    std::string out;
    while (i_ < s_.size() && s_[i_] != '"') {
      if (s_[i_] == '\\') {
        ++i_;
        if (i_ >= s_.size()) fail("bad escape");
        const char e = s_[i_];
        if (e == 'u') {
          out += "\\u";  // kept raw: no key the spike reads contains one
        } else {
          out += e == 'n' ? '\n' : e == 't' ? '\t' : e;
        }
        ++i_;
      } else {
        out += s_[i_++];
      }
    }
    expect('"');
    return out;
  }
  Json value() {
    Json v;
    const char c = peek();
    if (c == '{') {
      v.kind = Json::Kind::kObject;
      ++i_;
      if (peek() == '}') {
        ++i_;
        return v;
      }
      for (;;) {
        std::string k = string();
        expect(':');
        v.members.emplace_back(std::move(k), value());
        if (peek() == ',') {
          ++i_;
          continue;
        }
        expect('}');
        return v;
      }
    }
    if (c == '[') {
      v.kind = Json::Kind::kArray;
      ++i_;
      if (peek() == ']') {
        ++i_;
        return v;
      }
      for (;;) {
        v.items.push_back(value());
        if (peek() == ',') {
          ++i_;
          continue;
        }
        expect(']');
        return v;
      }
    }
    if (c == '"') {
      v.kind = Json::Kind::kString;
      v.str = string();
      return v;
    }
    if (s_.substr(i_, 4) == "true" || s_.substr(i_, 5) == "false") {
      v.kind = Json::Kind::kBool;
      v.boolean = s_[i_] == 't';
      i_ += v.boolean ? 4 : 5;
      return v;
    }
    if (s_.substr(i_, 4) == "null") {
      i_ += 4;
      return v;
    }
    const std::size_t start = i_;
    while (i_ < s_.size() && std::strchr("+-0123456789.eE", s_[i_]) != nullptr) ++i_;
    if (start == i_) fail("unexpected character");
    v.kind = Json::Kind::kNumber;
    v.number = std::strtod(std::string(s_.substr(start, i_ - start)).c_str(), nullptr);
    return v;
  }

  std::string_view s_;
  std::size_t i_ = 0;
};

std::string read_file(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p.string());
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Every tensor of a single-file safetensors checkpoint, converted to F32.
std::map<std::string, Tensor> load_safetensors(const fs::path& file) {
  const std::string bytes = read_file(file);
  std::uint64_t header_len = 0;
  std::memcpy(&header_len, bytes.data(), 8);
  const Json header = JsonParser(std::string_view(bytes).substr(8, header_len)).parse();
  const char* data = bytes.data() + 8 + header_len;
  std::map<std::string, Tensor> out;
  for (const auto& [name, e] : header.members) {
    if (name == "__metadata__") continue;
    std::vector<std::int64_t> dims;
    for (const Json& d : e.at("shape").items) dims.push_back(static_cast<std::int64_t>(d.number));
    const Shape shape(dims.data(), static_cast<int>(dims.size()));
    const auto begin = static_cast<std::size_t>(e.at("data_offsets").items[0].number);
    const std::string& dt = e.at("dtype").str;
    Tensor t = Tensor::empty(shape);
    float* dst = t.f32();
    if (dt == "BF16") {
      for (std::int64_t i = 0; i < shape.numel(); ++i) {
        std::uint16_t v;
        std::memcpy(&v, data + begin + 2 * static_cast<std::size_t>(i), 2);
        dst[i] = bf16_to_f32(v);
      }
    } else if (dt == "F32") {
      std::memcpy(dst, data + begin, 4 * static_cast<std::size_t>(shape.numel()));
    } else {
      throw std::runtime_error("unsupported dtype " + dt + " for " + name);
    }
    out.emplace(name, std::move(t));
  }
  return out;
}

struct Config {
  std::int64_t vocab, hidden, inter, layers, heads, kv_heads, head_dim;
  float eps, theta;
};

Config load_config(const fs::path& file) {
  const Json j = JsonParser(read_file(file)).parse();
  auto i = [&](const char* k) { return static_cast<std::int64_t>(j.at(k).number); };
  Config c{i("vocab_size"),          i("hidden_size"),         i("intermediate_size"),
           i("num_hidden_layers"),   i("num_attention_heads"), i("num_key_value_heads"),
           0,
           static_cast<float>(j.at("rms_norm_eps").number),
           static_cast<float>(j.at("rope_theta").number)};
  c.head_dim = c.hidden / c.heads;
  if (!j.at("tie_word_embeddings").boolean) throw std::runtime_error("spike assumes tied embeddings");
  return c;
}

// ===========================================================================
// The Q8_0 format
// ===========================================================================

constexpr int kQK = 32;  // weights per block

struct BlockQ8_0 {  // 34 bytes, byte-compatible with llama.cpp's block_q8_0
  std::uint16_t d;  // fp16 scale
  std::int8_t qs[kQK];
  float scale() const { return f16_hw(d); }
  void set_scale(float v) { d = f32_to_f16(v); }
};

struct BlockQ8_0F {  // 36 bytes: the same with an fp32 scale
  float d;
  std::int8_t qs[kQK];
  float scale() const { return d; }
  void set_scale(float v) { d = v; }
};

static_assert(sizeof(BlockQ8_0) == 34 && alignof(BlockQ8_0) == 2);
static_assert(sizeof(BlockQ8_0F) == 36);

enum class Rounding { kNearest, kTruncate };  // kTruncate is a deliberate bug, see quality

// THE reference quantizer: scalar, obviously correct, the definition every
// optimized quantizer must reproduce byte for byte.
//
//   amax = max |x| over the block
//   d    = amax / 127, stored (fp16: rounded to nearest even)
//   q    = clamp(round(x * (1 / d_stored)), -127, 127), ties away from zero
//
// It quantizes against the STORED scale, not the fp32 d, so the round-trip
// error is at most d_stored / 2 per element (llama.cpp's reference uses the
// unrounded d here and pays up to an extra 127 * |d - d_stored|). The clamp
// only ever bites when an fp16 d_stored is subnormal and rounded below d.
// d_stored below FLT_MIN (or 0) quantizes the block to all zeros, so 1/d never
// overflows.
template <class Block, Rounding R = Rounding::kNearest>
void quantize_ref(const float* x, Block* y, std::int64_t k) {
  for (std::int64_t b = 0; b < k / kQK; ++b) {
    const float* xb = x + b * kQK;
    float amax = 0.0f;
    for (int j = 0; j < kQK; ++j) amax = std::max(amax, std::fabs(xb[j]));
    y[b].set_scale(amax / 127.0f);
    const float d = y[b].scale();
    const float id = d >= std::numeric_limits<float>::min() ? 1.0f / d : 0.0f;
    for (int j = 0; j < kQK; ++j) {
      const float v = R == Rounding::kNearest ? std::round(xb[j] * id) : std::trunc(xb[j] * id);
      y[b].qs[j] = static_cast<std::int8_t>(std::clamp(v, -127.0f, 127.0f));
    }
  }
}

template <class Block>
void dequantize_row(const Block* y, float* x, std::int64_t k) {
  for (std::int64_t b = 0; b < k / kQK; ++b) {
    const float d = y[b].scale();
    for (int j = 0; j < kQK; ++j) x[b * kQK + j] = static_cast<float>(y[b].qs[j]) * d;
  }
}

// NEON activation quantizer (fp32 scale): the per-token step of w8a8. Must be
// byte-identical to quantize_ref<BlockQ8_0F>; selftest checks it.
void quantize_act(const float* x, BlockQ8_0F* y, std::int64_t k) {
  for (std::int64_t b = 0; b < k / kQK; ++b) {
    const float* xb = x + b * kQK;
    float32x4_t v[8];
    for (int i = 0; i < 8; ++i) v[i] = vld1q_f32(xb + 4 * i);
    const float32x4_t m0 = vmaxq_f32(vabsq_f32(v[0]), vabsq_f32(v[1]));
    const float32x4_t m1 = vmaxq_f32(vabsq_f32(v[2]), vabsq_f32(v[3]));
    const float32x4_t m2 = vmaxq_f32(vabsq_f32(v[4]), vabsq_f32(v[5]));
    const float32x4_t m3 = vmaxq_f32(vabsq_f32(v[6]), vabsq_f32(v[7]));
    const float amax = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));
    const float d = amax / 127.0f;
    y[b].d = d;
    const float id = d >= std::numeric_limits<float>::min() ? 1.0f / d : 0.0f;
    int16x8_t h[4];
    for (int i = 0; i < 4; ++i) {
      // vcvtaq rounds to nearest with ties away from zero, like std::round.
      int32x4_t lo = vcvtaq_s32_f32(vmulq_n_f32(v[2 * i], id));
      int32x4_t hi = vcvtaq_s32_f32(vmulq_n_f32(v[2 * i + 1], id));
      lo = vminq_s32(vmaxq_s32(lo, vdupq_n_s32(-127)), vdupq_n_s32(127));
      hi = vminq_s32(vmaxq_s32(hi, vdupq_n_s32(-127)), vdupq_n_s32(127));
      h[i] = vcombine_s16(vmovn_s32(lo), vmovn_s32(hi));
    }
    vst1q_s8(y[b].qs, vcombine_s8(vmovn_s16(h[0]), vmovn_s16(h[1])));
    vst1q_s8(y[b].qs + 16, vcombine_s8(vmovn_s16(h[2]), vmovn_s16(h[3])));
  }
}

// ===========================================================================
// Kernels. out[j] = sum_p x[p] * w[j, p] for j in [0, n), one activation row.
// ===========================================================================

// How a block's 32 int8 become 8 float32x4: through int32 (sxtl, sxtl, scvtf)
// or through fp16 (sxtl, scvtf .8h, fcvtl), which is exact for |q| <= 127 and
// does 8 lanes per conversion instead of 4.
enum class Widen { kViaInt32, kViaHalf };

template <Widen W>
inline void widen_block(const std::int8_t* qs, float32x4_t f[8]) {
  const int8x16_t q0 = vld1q_s8(qs);
  const int8x16_t q1 = vld1q_s8(qs + 16);
  const int16x8_t h[4] = {vmovl_s8(vget_low_s8(q0)), vmovl_high_s8(q0), vmovl_s8(vget_low_s8(q1)),
                          vmovl_high_s8(q1)};
  for (int i = 0; i < 4; ++i) {
    if constexpr (W == Widen::kViaInt32) {
      f[2 * i] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(h[i])));
      f[2 * i + 1] = vcvtq_f32_s32(vmovl_high_s16(h[i]));
    } else {
      const float16x8_t g = vcvtq_f16_s16(h[i]);
      f[2 * i] = vcvt_f32_f16(vget_low_f16(g));
      f[2 * i + 1] = vcvt_high_f32_f16(g);
    }
  }
}

// sum_j q[j] * x[j] over one block, unscaled, as 4 lanes.
template <Widen W>
inline float32x4_t block_dot(const std::int8_t* qs, const float32x4_t* xv) {
  float32x4_t f[8];
  widen_block<W>(qs, f);
  float32x4_t a = vmulq_f32(f[0], xv[0]);
  float32x4_t b = vmulq_f32(f[1], xv[1]);
  a = vfmaq_f32(a, f[2], xv[2]);
  b = vfmaq_f32(b, f[3], xv[3]);
  a = vfmaq_f32(a, f[4], xv[4]);
  b = vfmaq_f32(b, f[5], xv[5]);
  a = vfmaq_f32(a, f[6], xv[6]);
  b = vfmaq_f32(b, f[7], xv[7]);
  return vaddq_f32(a, b);
}

template <class Block, Widen W, int NR>
inline void dequant_rows(const float* x, const Block* w, float* out, std::int64_t nb) {
  float32x4_t acc[NR];
  for (int r = 0; r < NR; ++r) acc[r] = vdupq_n_f32(0.0f);
  for (std::int64_t b = 0; b < nb; ++b) {
    float32x4_t xv[8];
    for (int i = 0; i < 8; ++i) xv[i] = vld1q_f32(x + b * kQK + 4 * i);
    for (int r = 0; r < NR; ++r) {
      const Block& blk = w[r * nb + b];
      acc[r] = vfmaq_n_f32(acc[r], block_dot<W>(blk.qs, xv), blk.scale());
    }
  }
  for (int r = 0; r < NR; ++r) out[r] = vaddvq_f32(acc[r]);
}

template <class Block, Widen W>
void matvec_dequant(const float* x, const Block* w, float* out, std::int64_t n, std::int64_t k) {
  const std::int64_t nb = k / kQK;
  std::int64_t j = 0;
  for (; j + 4 <= n; j += 4) dequant_rows<Block, W, 4>(x, w + j * nb, out + j, nb);
  for (; j < n; ++j) dequant_rows<Block, W, 1>(x, w + j * nb, out + j, nb);
}

// Two accumulator sets along the blocks so the per-row FMA chain is not the
// limit once the weights are in cache.
template <class Block, int NR>
inline void w8a8_rows(const BlockQ8_0F* xq, const Block* w, float* out, std::int64_t nb) {
  float32x4_t acc[2][NR];
  for (int u = 0; u < 2; ++u)
    for (int r = 0; r < NR; ++r) acc[u][r] = vdupq_n_f32(0.0f);
  auto step = [&](std::int64_t b, int u) {
    const int8x16_t x0 = vld1q_s8(xq[b].qs);
    const int8x16_t x1 = vld1q_s8(xq[b].qs + 16);
    const float dx = xq[b].d;
    for (int r = 0; r < NR; ++r) {
      const Block& blk = w[r * nb + b];
      int32x4_t s = vdotq_s32(vdupq_n_s32(0), vld1q_s8(blk.qs), x0);
      s = vdotq_s32(s, vld1q_s8(blk.qs + 16), x1);
      acc[u][r] = vfmaq_n_f32(acc[u][r], vcvtq_f32_s32(s), blk.scale() * dx);
    }
  };
  std::int64_t b = 0;
  for (; b + 2 <= nb; b += 2) {
    step(b, 0);
    step(b + 1, 1);
  }
  if (b < nb) step(b, 0);
  for (int r = 0; r < NR; ++r) out[r] = vaddvq_f32(vaddq_f32(acc[0][r], acc[1][r]));
}

template <class Block>
void matvec_w8a8(const BlockQ8_0F* xq, const Block* w, float* out, std::int64_t n,
                 std::int64_t k) {
  const std::int64_t nb = k / kQK;
  std::int64_t j = 0;
  for (; j + 4 <= n; j += 4) w8a8_rows<Block, 4>(xq, w + j * nb, out + j, nb);
  for (; j < n; ++j) w8a8_rows<Block, 1>(xq, w + j * nb, out + j, nb);
}

// For comparison, not Q8_0: the checkpoint's own bf16 values, widened to fp32
// with a shift (exact) and multiplied in fp32. Half the bytes of f32 and no
// accuracy change at all.
template <int NR>
inline void bf16_rows(const float* x, const std::uint16_t* w, float* out, std::int64_t k) {
  float32x4_t acc[2][NR];
  for (int u = 0; u < 2; ++u)
    for (int r = 0; r < NR; ++r) acc[u][r] = vdupq_n_f32(0.0f);
  for (std::int64_t p = 0; p + 8 <= k; p += 8) {
    const float32x4_t x0 = vld1q_f32(x + p);
    const float32x4_t x1 = vld1q_f32(x + p + 4);
    for (int r = 0; r < NR; ++r) {
      const uint16x8_t v = vld1q_u16(w + r * k + p);
      acc[0][r] = vfmaq_f32(acc[0][r], vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v), 16)), x0);
      acc[1][r] = vfmaq_f32(acc[1][r], vreinterpretq_f32_u32(vshll_high_n_u16(v, 16)), x1);
    }
  }
  for (int r = 0; r < NR; ++r) out[r] = vaddvq_f32(vaddq_f32(acc[0][r], acc[1][r]));
}

void matvec_bf16(const float* x, const std::uint16_t* w, float* out, std::int64_t n, std::int64_t k) {
  std::int64_t j = 0;
  for (; j + 4 <= n; j += 4) bf16_rows<4>(x, w + j * k, out + j, k);
  for (; j < n; ++j) bf16_rows<1>(x, w + j * k, out + j, k);
}

// ===========================================================================
// Weights
// ===========================================================================

struct Linear {
  std::string name;
  std::int64_t n = 0, k = 0;
  Tensor f32;                    // [n, k], the checkpoint values
  std::vector<BlockQ8_0> q16;    // [n, k / 32], fp16 scales
  std::vector<BlockQ8_0F> q32;   // [n, k / 32], fp32 scales
  std::vector<BlockQ8_0> qbug;   // fp16 scales, truncating quantizer (the bug)
  std::vector<std::uint16_t> bf16;  // [n, k], the checkpoint's bf16 bits
  Tensor dq16;                   // [n, k] f32 = dequantize(q16): the Q8_0 model as fp32

  void quantize() {
    const std::int64_t nb = n * k / kQK;
    q16.resize(nb);
    q32.resize(nb);
    qbug.resize(nb);
    quantize_ref(f32.f32(), q16.data(), n * k);
    quantize_ref(f32.f32(), q32.data(), n * k);
    quantize_ref<BlockQ8_0, Rounding::kTruncate>(f32.f32(), qbug.data(), n * k);
    bf16.resize(static_cast<std::size_t>(n * k));
    for (std::int64_t i = 0; i < n * k; ++i) bf16[static_cast<std::size_t>(i)] = f32_to_bf16(f32.f32()[i]);
    dq16 = Tensor::empty({n, k});
    dequantize_row(q16.data(), dq16.f32(), n * k);
  }
  std::size_t bytes_f32() const { return static_cast<std::size_t>(n * k) * 4; }
  std::size_t bytes_bf16() const { return static_cast<std::size_t>(n * k) * 2; }
  std::size_t bytes_q16() const { return static_cast<std::size_t>(n * k / kQK) * sizeof(BlockQ8_0); }
  std::size_t bytes_q32() const { return static_cast<std::size_t>(n * k / kQK) * sizeof(BlockQ8_0F); }
};

struct Layer {
  Tensor in_norm, post_norm;
  Linear q, k, v, o, gate, up, down;
  Linear* all() { return &q; }  // the seven, in declaration order
};

struct ModelWeights {
  Config cfg{};
  Linear embed;  // also the lm_head
  Tensor final_norm;
  std::vector<Layer> layers;

  std::vector<Linear*> projections() {
    std::vector<Linear*> v;
    for (Layer& l : layers)
      for (Linear* p : {&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down}) v.push_back(p);
    return v;
  }
};

ModelWeights load_model(const fs::path& dir) {
  ModelWeights w;
  w.cfg = load_config(dir / "config.json");
  auto ts = load_safetensors(dir / "model.safetensors");
  auto take = [&](const std::string& name) {
    auto it = ts.find(name);
    if (it == ts.end()) throw std::runtime_error("missing tensor " + name);
    return it->second;
  };
  auto linear = [&](Linear& l, const std::string& name) {
    l.name = name;
    l.f32 = take(name);
    l.n = l.f32.dim(0);
    l.k = l.f32.dim(1);
    if (l.k % kQK != 0) throw std::runtime_error(name + ": k not a multiple of 32");
    l.quantize();
  };
  linear(w.embed, "model.embed_tokens.weight");
  w.final_norm = take("model.norm.weight");
  w.layers.resize(static_cast<std::size_t>(w.cfg.layers));
  for (std::int64_t i = 0; i < w.cfg.layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    Layer& l = w.layers[static_cast<std::size_t>(i)];
    l.in_norm = take(p + "input_layernorm.weight");
    l.post_norm = take(p + "post_attention_layernorm.weight");
    linear(l.q, p + "self_attn.q_proj.weight");
    linear(l.k, p + "self_attn.k_proj.weight");
    linear(l.v, p + "self_attn.v_proj.weight");
    linear(l.o, p + "self_attn.o_proj.weight");
    linear(l.gate, p + "mlp.gate_proj.weight");
    linear(l.up, p + "mlp.up_proj.weight");
    linear(l.down, p + "mlp.down_proj.weight");
  }
  return w;
}

// ===========================================================================
// One interface over every kernel, for the benchmarks
// ===========================================================================

enum class Kind {
  kF32,          // ops::matmul, the engine's kernel today
  kBF16,         // bf16 weights widened to fp32 (not Q8_0; the exact alternative)
  kDqI32_S16,    // dequant via int32, fp16 scales
  kDqF16_S16,    // dequant via fp16, fp16 scales
  kDqF16_S32,    // dequant via fp16, fp32 scales
  kW8A8_S16,     // int8 dot, fp16 weight scales
  kW8A8_S32,     // int8 dot, fp32 weight scales
};

const char* kind_name(Kind k) {
  switch (k) {
    case Kind::kF32: return "f32 ops::matmul";
    case Kind::kBF16: return "bf16 -> f32";
    case Kind::kDqI32_S16: return "dequant/i32  34B";
    case Kind::kDqF16_S16: return "dequant/f16  34B";
    case Kind::kDqF16_S32: return "dequant/f16  36B";
    case Kind::kW8A8_S16: return "w8a8 sdot    34B";
    case Kind::kW8A8_S32: return "w8a8 sdot    36B";
  }
  return "?";
}

std::size_t kind_bytes(Kind kind, const Linear& l) {
  switch (kind) {
    case Kind::kF32: return l.bytes_f32();
    case Kind::kBF16: return l.bytes_bf16();
    case Kind::kDqF16_S32:
    case Kind::kW8A8_S32: return l.bytes_q32();
    default: return l.bytes_q16();
  }
}

// Rows [r0, r1) of out = x · Wᵀ. xq is scratch for k / 32 activation blocks.
void run_rows(Kind kind, const Linear& l, const float* x, BlockQ8_0F* xq, float* out,
              std::int64_t r0, std::int64_t r1) {
  const std::int64_t nb = l.k / kQK;
  const std::int64_t n = r1 - r0;
  switch (kind) {
    case Kind::kF32: {
      const Tensor xt = Tensor::wrap(const_cast<float*>(x), {1, l.k});
      const Tensor wt = Tensor::wrap(const_cast<float*>(l.f32.f32() + r0 * l.k), {n, l.k});
      Tensor ot = Tensor::wrap(out + r0, {1, n});
      ops::matmul(xt, wt, ot);
      break;
    }
    case Kind::kBF16:
      matvec_bf16(x, l.bf16.data() + r0 * l.k, out + r0, n, l.k);
      break;
    case Kind::kDqI32_S16:
      matvec_dequant<BlockQ8_0, Widen::kViaInt32>(x, l.q16.data() + r0 * nb, out + r0, n, l.k);
      break;
    case Kind::kDqF16_S16:
      matvec_dequant<BlockQ8_0, Widen::kViaHalf>(x, l.q16.data() + r0 * nb, out + r0, n, l.k);
      break;
    case Kind::kDqF16_S32:
      matvec_dequant<BlockQ8_0F, Widen::kViaHalf>(x, l.q32.data() + r0 * nb, out + r0, n, l.k);
      break;
    case Kind::kW8A8_S16:
      quantize_act(x, xq, l.k);
      matvec_w8a8(xq, l.q16.data() + r0 * nb, out + r0, n, l.k);
      break;
    case Kind::kW8A8_S32:
      quantize_act(x, xq, l.k);
      matvec_w8a8(xq, l.q32.data() + r0 * nb, out + r0, n, l.k);
      break;
  }
}

// ===========================================================================
// selftest: the conversions, the quantizer and the kernels are right
// ===========================================================================

struct ErrStats {
  double max_abs = 0, max_rel = 0;
  bool within = true;
  void add(double actual, double expected, const test::Tolerance& t) {
    const double e = std::fabs(actual - expected);
    max_abs = std::max(max_abs, e);
    if (expected != 0) max_rel = std::max(max_rel, e / std::fabs(expected));
    if (e > t.atol + t.rtol * std::fabs(expected)) within = false;
  }
};

void selftest_f16() {
  // Every non-negative float up to and including +inf: a scale is never
  // negative, and the sign bit is a plain OR.
  std::uint64_t mismatches = 0;
  const double t0 = now_s();
  for (std::uint32_t u = 0; u <= 0x7f800000u; ++u) {
    const float f = std::bit_cast<float>(u);
    const auto hw = std::bit_cast<std::uint16_t>(static_cast<_Float16>(f));
    if (f32_to_f16(f) != hw) ++mismatches;
  }
  // And f16 -> f32 -> f16 for all 65536 patterns, against tensor.h.
  std::uint64_t roundtrip_bad = 0;
  for (std::uint32_t h = 0; h < 65536; ++h) {
    const auto hb = static_cast<std::uint16_t>(h);
    const float f = f16_to_f32(hb);
    if (std::isnan(f)) continue;
    if (f16_hw(hb) != f || f32_to_f16(f) != hb) ++roundtrip_bad;
  }
  std::printf("f32_to_f16 vs hardware fcvt, all 2^31 non-negative floats: %llu mismatches (%.1f s)\n",
              static_cast<unsigned long long>(mismatches), now_s() - t0);
  std::printf("f16 -> f32 -> f16 over all non-NaN f16: %llu mismatches\n",
              static_cast<unsigned long long>(roundtrip_bad));
}

template <class Block>
void roundtrip_stats(const char* label, std::vector<Linear*> ls,
                     std::vector<Block> Linear::*member) {
  std::uint64_t blocks = 0, zero_scale = 0, subnormal_scale = 0, clamped = 0, bound_bad = 0;
  double max_err_over_d = 0, max_err_over_amax254 = 0, sq_err = 0, sq_ref = 0;
  double min_d = 1e30, max_d = 0;
  for (Linear* l : ls) {
    const float* x = l->f32.f32();
    const std::vector<Block>& q = l->*member;
    for (std::size_t b = 0; b < q.size(); ++b) {
      ++blocks;
      const double d = q[b].scale();
      float amax = 0;
      for (int j = 0; j < kQK; ++j) amax = std::max(amax, std::fabs(x[b * kQK + j]));
      if (d == 0) ++zero_scale;
      if (d > 0) min_d = std::min(min_d, d);
      max_d = std::max(max_d, d);
      if constexpr (std::is_same_v<Block, BlockQ8_0>) {
        if (d != 0 && d < 6.103515625e-05) ++subnormal_scale;  // 2^-14
      }
      for (int j = 0; j < kQK; ++j) {
        const double xv = x[b * kQK + j];
        const double err = std::fabs(xv - q[b].qs[j] * d);
        sq_err += err * err;
        sq_ref += xv * xv;
        if (std::fabs(xv) > 127.5 * d) {
          ++clamped;  // only possible when d was rounded down; checked separately
          continue;
        }
        if (d > 0) {
          max_err_over_d = std::max(max_err_over_d, err / d);
          if (err > d * (0.5 + 0x1p-15)) ++bound_bad;
        }
        if (amax > 0) max_err_over_amax254 = std::max(max_err_over_amax254, err / (amax / 254.0));
      }
    }
  }
  std::printf(
      "  %-26s blocks %9llu  d in [%.3e, %.3e]  zero-d %llu  subnormal-d %llu  clamped %llu\n"
      "  %-26s max |x - q*d|/d %.6f (bound 0.5 + 2^-15: %llu violations)  "
      "max err/(amax/254) %.6f  rel Frobenius err %.3e\n",
      label, static_cast<unsigned long long>(blocks), min_d, max_d,
      static_cast<unsigned long long>(zero_scale),
      static_cast<unsigned long long>(subnormal_scale), static_cast<unsigned long long>(clamped), "",
      max_err_over_d, static_cast<unsigned long long>(bound_bad), max_err_over_amax254,
      std::sqrt(sq_err / sq_ref));
}

void selftest(ModelWeights& w) {
  std::printf("\n== selftest ==\n");
  selftest_f16();

  std::vector<Linear*> proj = w.projections();
  std::vector<Linear*> all = proj;
  all.push_back(&w.embed);

  // Weight values: finite, and how far each tensor type is from the f16 range.
  std::uint64_t nonfinite = 0;
  for (Linear* l : all)
    for (std::int64_t i = 0; i < l->n * l->k; ++i)
      if (!std::isfinite(l->f32.f32()[i])) ++nonfinite;
  std::printf("non-finite weights: %llu\n", static_cast<unsigned long long>(nonfinite));

  std::printf("round trip, reference quantizer, real weights:\n");
  roundtrip_stats<BlockQ8_0>("projections, fp16 d", proj, &Linear::q16);
  roundtrip_stats<BlockQ8_0F>("projections, fp32 d", proj, &Linear::q32);
  roundtrip_stats<BlockQ8_0>("embed_tokens, fp16 d", {&w.embed}, &Linear::q16);
  roundtrip_stats<BlockQ8_0F>("embed_tokens, fp32 d", {&w.embed}, &Linear::q32);
  roundtrip_stats<BlockQ8_0>("projections, TRUNC bug", proj, &Linear::qbug);
  {
    // Per tensor type, fp16 scales: which projections lose the most.
    const char* names[7] = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                            "down_proj"};
    for (int t = 0; t < 7; ++t) {
      std::vector<Linear*> ls;
      for (Layer& l : w.layers) ls.push_back(l.all() + t);
      roundtrip_stats<BlockQ8_0>(names[t], ls, &Linear::q16);
    }
  }

  // Optimized activation quantizer == reference, byte for byte, on real
  // weight rows and on N(0, s) rows over many scales.
  {
    std::mt19937 rng(7);
    std::uint64_t rows = 0, bad = 0;
    std::vector<BlockQ8_0F> a(48), b(48);
    std::vector<float> x(1536);
    for (int it = 0; it < 20000; ++it) {
      const float s = std::ldexp(1.0f, static_cast<int>(rng() % 60) - 40);
      std::normal_distribution<float> nd(0.0f, s);
      for (float& v : x) v = nd(rng);
      if (it % 7 == 0) x[rng() % 1536] = 0.0f;
      if (it % 11 == 0) std::fill(x.begin(), x.begin() + 32, 0.0f);  // an all-zero block
      quantize_ref(x.data(), a.data(), 1536);
      quantize_act(x.data(), b.data(), 1536);
      ++rows;
      if (std::memcmp(a.data(), b.data(), a.size() * sizeof(BlockQ8_0F)) != 0) ++bad;
    }
    for (Linear* l : proj) {
      for (std::int64_t r = 0; r < l->n; r += 97) {
        std::vector<BlockQ8_0F> ra(static_cast<std::size_t>(l->k / kQK)), rb(ra.size());
        quantize_ref(l->f32.f32() + r * l->k, ra.data(), l->k);
        quantize_act(l->f32.f32() + r * l->k, rb.data(), l->k);
        ++rows;
        if (std::memcmp(ra.data(), rb.data(), ra.size() * sizeof(BlockQ8_0F)) != 0) ++bad;
      }
    }
    std::printf("NEON activation quantizer vs reference: %llu rows, %llu differ\n",
                static_cast<unsigned long long>(rows), static_cast<unsigned long long>(bad));
  }

  // Kernels against ref::matmul on dequantized weights, decode shapes, real
  // weights, N(0, 1) activations. tol::kMatmul is the existing matmul
  // tolerance; the question is whether the Q8_0 kernels fit inside it.
  std::printf("kernels vs ref::matmul(x, dequantize(W)) [w8a8: ref on dequantize(quantize(x))]\n");
  std::mt19937 rng(42);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const std::vector<std::pair<const char*, Linear*>> shapes = {
      {"[1,576]x[1536,576]^T", &w.layers[3].gate},
      {"[1,1536]x[576,1536]^T", &w.layers[3].down},
      {"[1,576]x[49152,576]^T", &w.embed}};
  for (const auto& [label, l] : shapes) {
    std::vector<float> x(static_cast<std::size_t>(l->k));
    for (float& v : x) v = nd(rng);
    std::vector<BlockQ8_0F> xq(static_cast<std::size_t>(l->k / kQK));
    quantize_ref(x.data(), xq.data(), l->k);
    std::vector<float> xdq(x.size());
    dequantize_row(xq.data(), xdq.data(), l->k);

    Tensor w16 = Tensor::empty({l->n, l->k}), w32 = Tensor::empty({l->n, l->k});
    dequantize_row(l->q16.data(), w16.f32(), l->n * l->k);
    dequantize_row(l->q32.data(), w32.f32(), l->n * l->k);
    auto ref = [&](const std::vector<float>& xin, const Tensor& wt) {
      Tensor xt = Tensor::wrap(const_cast<float*>(xin.data()), {1, l->k});
      Tensor ot = Tensor::empty({1, l->n});
      ref::matmul(xt, wt, ot);
      return ot;
    };
    const Tensor e16 = ref(x, w16), e32 = ref(x, w32), e16a = ref(xdq, w16), e32a = ref(xdq, w32);
    const Tensor ef = ref(x, l->f32);
    for (Kind kind : {Kind::kF32, Kind::kBF16, Kind::kDqI32_S16, Kind::kDqF16_S16,
                      Kind::kDqF16_S32, Kind::kW8A8_S16, Kind::kW8A8_S32}) {
      std::vector<float> out(static_cast<std::size_t>(l->n));
      std::vector<BlockQ8_0F> scratch(xq.size());
      run_rows(kind, *l, x.data(), scratch.data(), out.data(), 0, l->n);
      const Tensor& e = kind == Kind::kF32 || kind == Kind::kBF16 ? ef
                        : kind == Kind::kDqI32_S16 || kind == Kind::kDqF16_S16 ? e16
                        : kind == Kind::kDqF16_S32 ? e32
                        : kind == Kind::kW8A8_S16  ? e16a
                                                   : e32a;
      ErrStats s;
      ErrStats vs_f32;  // the quantization error itself, for scale
      for (std::int64_t j = 0; j < l->n; ++j) {
        s.add(out[j], e.f32()[j], test::tol::kMatmul);
        vs_f32.add(out[j], ef.f32()[j], test::tol::kMatmul);
      }
      std::printf("  %-22s %-17s vs its ref: max abs %.2e rel %.2e %s | vs fp32 W: max abs %.2e\n",
                  label, kind_name(kind), s.max_abs, s.max_rel,
                  s.within ? "within kMatmul" : "OUTSIDE kMatmul", vs_f32.max_abs);
    }
  }
}

// ===========================================================================
// bench: decode shapes, one thread, interleaved min-of-N
// ===========================================================================

struct Bench {
  std::vector<float> x576, x1536, out;
  std::vector<BlockQ8_0F> xq;

  Bench() : x576(576), x1536(1536), out(49152), xq(1536 / kQK) {
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : x576) v = nd(rng);
    for (float& v : x1536) v = nd(rng);
  }
  const float* x_for(const Linear& l) const { return l.k == 576 ? x576.data() : x1536.data(); }

  // One pass over the list, every matrix once. Returns seconds.
  double sweep(Kind kind, const std::vector<const Linear*>& ls) {
    const double t0 = now_s();
    for (const Linear* l : ls) run_rows(kind, *l, x_for(*l), xq.data(), out.data(), 0, l->n);
    return now_s() - t0;
  }
};

struct Row {
  Kind kind;
  double best = 1e30;
  std::size_t bytes = 0;
};

// Interleaves the kinds rep by rep and keeps each one's minimum.
std::vector<Row> measure(Bench& b, const std::vector<Kind>& kinds,
                         const std::vector<const Linear*>& ls, int reps) {
  std::vector<Row> rows;
  for (Kind k : kinds) {
    Row r{k};
    for (const Linear* l : ls) r.bytes += kind_bytes(k, *l);
    rows.push_back(r);
  }
  for (Row& r : rows) b.sweep(r.kind, ls);  // warm-up
  for (int rep = 0; rep < reps; ++rep)
    for (Row& r : rows) r.best = std::min(r.best, b.sweep(r.kind, ls));
  return rows;
}

void print_rows(const char* title, const std::vector<Row>& rows, std::size_t per_sweep) {
  std::printf("\n%s\n", title);
  std::printf("  %-18s %12s %10s %12s %12s\n", "kernel", "us/matmul", "GB/s", "time/f32", "speed-up");
  const double f32 = rows[0].best;
  for (const Row& r : rows) {
    std::printf("  %-18s %12.1f %10.1f %12.3f %11.2fx\n", kind_name(r.kind),
                1e6 * r.best / static_cast<double>(per_sweep),
                static_cast<double>(r.bytes) / r.best * 1e-9, r.best / f32, f32 / r.best);
  }
}

// Plain sequential read, 4 NEON accumulators: the bandwidth ceiling.
double read_sweep(const float* p, std::int64_t n, float* sink) {
  const double t0 = now_s();
  float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
  for (std::int64_t i = 0; i + 16 <= n; i += 16) {
    a0 = vaddq_f32(a0, vld1q_f32(p + i));
    a1 = vaddq_f32(a1, vld1q_f32(p + i + 4));
    a2 = vaddq_f32(a2, vld1q_f32(p + i + 8));
    a3 = vaddq_f32(a3, vld1q_f32(p + i + 12));
  }
  *sink += vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
  return now_s() - t0;
}

void bench(ModelWeights& w) {
  std::printf("\n== bench (single thread) == %s\n", load_string().c_str());
  Bench b;
  const std::vector<Kind> kinds = {Kind::kF32,       Kind::kBF16,      Kind::kDqI32_S16,
                                   Kind::kDqF16_S16, Kind::kDqF16_S32, Kind::kW8A8_S16,
                                   Kind::kW8A8_S32};

  // Streaming from DRAM: every matrix of that shape in the model, so the
  // sweep (56-212 MB) is far larger than L2 (16 MB) plus the SLC, exactly as
  // in decode, where nothing a token reads is still cached from the last one.
  std::vector<const Linear*> s1, s2, s3 = {&w.embed};
  for (const Layer& l : w.layers) {
    s1.push_back(&l.gate);
    s1.push_back(&l.up);
    s2.push_back(&l.down);
  }
  const int reps = 25;
  print_rows("[1, 576] x [1536, 576]^T  (gate+up of all 30 layers per sweep, DRAM)",
             measure(b, kinds, s1, reps), s1.size());
  print_rows("[1, 1536] x [576, 1536]^T  (down of all 30 layers per sweep, DRAM)",
             measure(b, kinds, s2, reps), s2.size());
  print_rows("[1, 576] x [49152, 576]^T  lm_head (DRAM)", measure(b, kinds, s3, reps), 1);

  // Cache-resident: one matrix 32 times. This is the kernel's own ceiling,
  // what it would do if bandwidth were free.
  std::vector<const Linear*> h1(32, &w.layers[3].gate), h2(32, &w.layers[3].down);
  print_rows("[1, 576] x [1536, 576]^T  one matrix repeated (L2-resident: compute ceiling)",
             measure(b, kinds, h1, reps), h1.size());
  print_rows("[1, 1536] x [576, 1536]^T  one matrix repeated (L2-resident)",
             measure(b, kinds, h2, reps), h2.size());

  // A whole decode token's linear layers: 7 x 30 projections + lm_head.
  std::vector<const Linear*> token;
  for (const Layer& l : w.layers)
    for (const Linear* p : {&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down}) token.push_back(p);
  std::vector<const Linear*> token_lin = token;
  token.push_back(&w.embed);
  print_rows("one decode token, 210 projections + lm_head, all in the kernel's format",
             measure(b, kinds, token, 15), token.size());
  print_rows("one decode token, 210 projections only (lm_head excluded)",
             measure(b, kinds, token_lin, 15), token_lin.size());

  // Prefill, m = 64: ops::matmul runs its 4 x 4 register tiles; the spike's
  // Q8_0 kernels are m = 1 kernels called once per row (each matrix stays in
  // L2 across the 64 rows). Not a candidate design, just the cost of shipping
  // decode kernels only.
  {
    const std::int64_t m = 64;
    std::vector<float> X(static_cast<std::size_t>(m * 576)), O(static_cast<std::size_t>(m * 1536));
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : X) v = nd(rng);
    auto pass = [&](Kind kind) {
      const double t0 = now_s();
      for (const Linear* l : s1) {
        if (kind == Kind::kF32) {
          Tensor xt = Tensor::wrap(X.data(), {m, 576}), ot = Tensor::wrap(O.data(), {m, 1536});
          ops::matmul(xt, l->f32, ot);
        } else {
          for (std::int64_t i = 0; i < m; ++i)
            run_rows(kind, *l, X.data() + i * 576, b.xq.data(), O.data() + i * 1536, 0, l->n);
        }
      }
      return now_s() - t0;
    };
    const std::vector<Kind> pk = {Kind::kF32, Kind::kBF16, Kind::kDqF16_S16, Kind::kW8A8_S16};
    std::vector<double> best(pk.size(), 1e30);
    for (Kind k : pk) pass(k);
    for (int rep = 0; rep < 5; ++rep)
      for (std::size_t i = 0; i < pk.size(); ++i) best[i] = std::min(best[i], pass(pk[i]));
    std::printf("\nprefill m = 64, [64, 576] x [1536, 576]^T over gate+up of 30 layers (per-row Q8_0 kernels)\n");
    std::printf("  %-18s %12s %12s\n", "kernel", "ms/sweep", "time/f32");
    for (std::size_t i = 0; i < pk.size(); ++i)
      std::printf("  %-18s %12.2f %12.3f\n", kind_name(pk[i]), 1e3 * best[i], best[i] / best[0]);
  }

  // Read ceiling.
  std::vector<float> big(64ull << 20);  // 256 MB
  for (std::size_t i = 0; i < big.size(); i += 1024) big[i] = 1.0f;
  float sink = 0;
  double best = 1e30;
  for (int rep = 0; rep < 10; ++rep)
    best = std::min(best, read_sweep(big.data(), static_cast<std::int64_t>(big.size()), &sink));
  std::printf("\nplain sequential read, 256 MB, one thread: %.1f GB/s (sink %g)\n",
              static_cast<double>(big.size() * 4) / best * 1e-9, static_cast<double>(sink));
  std::printf("%s\n", load_string().c_str());
}

// ===========================================================================
// threads: the same decode token split over T threads, as a threaded engine
// would run it (row ranges per matmul, a barrier after each)
// ===========================================================================

class Team {
 public:
  explicit Team(int n) : n_(n) {
    for (int t = 1; t < n; ++t) workers_.emplace_back([this, t] { worker(t); });
  }
  ~Team() {
    stop_.store(true);
    go_.fetch_add(1, std::memory_order_release);
    for (std::thread& w : workers_) w.join();
  }
  void run(std::function<void(int)> fn) {
    job_ = std::move(fn);
    pending_.store(n_ - 1, std::memory_order_relaxed);
    go_.fetch_add(1, std::memory_order_release);
    job_(0);
    wait_while([&] { return pending_.load(std::memory_order_acquire) != 0; });
  }
  void barrier() {
    const unsigned gen = gen_.load(std::memory_order_acquire);
    if (count_.fetch_add(1, std::memory_order_acq_rel) == n_ - 1) {
      count_.store(0, std::memory_order_relaxed);
      gen_.fetch_add(1, std::memory_order_release);
    } else {
      wait_while([&] { return gen_.load(std::memory_order_acquire) == gen; });
    }
  }

 private:
  template <class F>
  static void wait_while(F cond) {
    for (int spins = 0; cond();)
      if (++spins > 2000) std::this_thread::yield();
  }
  void worker(int t) {
    unsigned seen = 0;
    for (;;) {
      wait_while([&] { return go_.load(std::memory_order_acquire) == seen; });
      seen = go_.load(std::memory_order_acquire);
      if (stop_.load()) return;
      job_(t);
      pending_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  int n_;
  std::vector<std::thread> workers_;
  std::function<void(int)> job_;
  std::atomic<unsigned> go_{0}, gen_{0};
  std::atomic<int> pending_{0}, count_{0};
  std::atomic<bool> stop_{false};
};

void threads(ModelWeights& w) {
  std::printf("\n== threads: one decode token (210 projections + lm_head) over T threads == %s\n",
              load_string().c_str());
  std::vector<const Linear*> token;
  for (const Layer& l : w.layers)
    for (const Linear* p : {&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down}) token.push_back(p);
  token.push_back(&w.embed);
  const std::vector<Kind> kinds = {Kind::kF32, Kind::kBF16, Kind::kDqF16_S16, Kind::kW8A8_S16};

  std::vector<float> x576(576), x1536(1536), out(49152);
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (float& v : x576) v = nd(rng);
  for (float& v : x1536) v = nd(rng);
  std::vector<float> big(64ull << 20);
  for (std::size_t i = 0; i < big.size(); i += 1024) big[i] = 1.0f;

  std::printf("  %3s  %-18s %10s %10s %10s\n", "T", "kernel", "ms/token", "GB/s", "vs f32");
  for (int T : {1, 2, 3, 4, 6, 8}) {
    Team team(T);
    std::vector<std::vector<BlockQ8_0F>> xq(static_cast<std::size_t>(T),
                                            std::vector<BlockQ8_0F>(1536 / kQK));
    auto token_pass = [&](Kind kind) {
      const double t0 = now_s();
      team.run([&](int t) {
        for (const Linear* l : token) {
          // 4-row-aligned contiguous chunk per thread.
          const std::int64_t chunk = ((l->n + T - 1) / T + 3) / 4 * 4;
          const std::int64_t r0 = std::min<std::int64_t>(l->n, t * chunk);
          const std::int64_t r1 = std::min<std::int64_t>(l->n, r0 + chunk);
          if (r0 < r1)
            run_rows(kind, *l, l->k == 576 ? x576.data() : x1536.data(),
                     xq[static_cast<std::size_t>(t)].data(), out.data(), r0, r1);
          team.barrier();
        }
      });
      return now_s() - t0;
    };
    auto read_pass = [&] {
      std::vector<float> sinks(static_cast<std::size_t>(T), 0.0f);
      const double t0 = now_s();
      team.run([&](int t) {
        const std::int64_t n = static_cast<std::int64_t>(big.size()) / T;
        read_sweep(big.data() + t * n, n, &sinks[static_cast<std::size_t>(t)]);
      });
      return now_s() - t0;
    };
    std::vector<double> best(kinds.size(), 1e30);
    double best_read = 1e30;
    for (Kind k : kinds) token_pass(k);
    for (int rep = 0; rep < 40; ++rep) {
      for (std::size_t i = 0; i < kinds.size(); ++i) best[i] = std::min(best[i], token_pass(kinds[i]));
      best_read = std::min(best_read, read_pass());
    }
    for (std::size_t i = 0; i < kinds.size(); ++i) {
      std::size_t bytes = 0;
      for (const Linear* l : token) bytes += kind_bytes(kinds[i], *l);
      std::printf("  %3d  %-18s %10.2f %10.1f %9.2fx\n", T, kind_name(kinds[i]), 1e3 * best[i],
                  static_cast<double>(bytes) / best[i] * 1e-9, best[0] / best[i]);
    }
    std::printf("  %3d  %-18s %10s %10.1f\n", T, "plain read 256 MB", "",
                static_cast<double>(big.size() * 4) / best_read * 1e-9);
  }
  std::printf("%s\n", load_string().c_str());
}

// ===========================================================================
// quality: a forward pass with each linear layer swappable
// ===========================================================================

// kRefDequant / kRefW8A8 are the Q8_0 path's REFERENCE: ops::matmul (already
// proven against ref::matmul and torch) on dequantize(W), and for w8a8 on
// dequantize(quantize_ref(x)) too. The Q8_0 kernels must reproduce these.
enum class Kern { kF32, kDequant, kW8A8, kRefDequant, kRefW8A8 };
enum class Weights8 { kQ16, kQ32, kBug };

struct Variant {
  std::string name;
  Kern linear = Kern::kF32;  // the 7 projections
  Kern head = Kern::kF32;    // lm_head
  bool embed_q8 = false;     // embedding lookup from the quantized table
  Weights8 wq = Weights8::kQ16;
};

void linear(const Tensor& x, const Linear& l, Kern kern, Weights8 wq, Tensor& out) {
  if (kern == Kern::kF32) {
    ops::matmul(x, l.f32, out);
    return;
  }
  const std::int64_t m = x.dim(0);
  if (kern == Kern::kRefDequant || kern == Kern::kRefW8A8) {
    if (wq != Weights8::kQ16) throw std::logic_error("reference kernels are fp16-scale only");
    if (kern == Kern::kRefDequant) {
      ops::matmul(x, l.dq16, out);
      return;
    }
    Tensor xd = Tensor::empty({m, l.k});
    std::vector<BlockQ8_0F> xq(static_cast<std::size_t>(l.k / kQK));
    for (std::int64_t i = 0; i < m; ++i) {
      quantize_ref(x.f32() + i * l.k, xq.data(), l.k);
      dequantize_row(xq.data(), xd.f32() + i * l.k, l.k);
    }
    ops::matmul(xd, l.dq16, out);
    return;
  }
  std::vector<BlockQ8_0F> xq(static_cast<std::size_t>(l.k / kQK));
  for (std::int64_t i = 0; i < m; ++i) {
    const float* xr = x.f32() + i * l.k;
    float* o = out.f32() + i * l.n;
    if (kern == Kern::kDequant) {
      if (wq == Weights8::kQ32)
        matvec_dequant<BlockQ8_0F, Widen::kViaHalf>(xr, l.q32.data(), o, l.n, l.k);
      else
        matvec_dequant<BlockQ8_0, Widen::kViaHalf>(xr, (wq == Weights8::kBug ? l.qbug : l.q16).data(),
                                                   o, l.n, l.k);
    } else {
      quantize_act(xr, xq.data(), l.k);
      if (wq == Weights8::kQ32)
        matvec_w8a8(xq.data(), l.q32.data(), o, l.n, l.k);
      else
        matvec_w8a8(xq.data(), (wq == Weights8::kBug ? l.qbug : l.q16).data(), o, l.n, l.k);
    }
  }
}

class Runner {
 public:
  Runner(const ModelWeights& w, Variant v, std::int64_t max_seq) : w_(w), v_(std::move(v)) {
    const Config& c = w.cfg;
    for (std::int64_t l = 0; l < c.layers; ++l) {
      kc_.push_back(Tensor::zeros({max_seq, c.kv_heads, c.head_dim}));
      vc_.push_back(Tensor::zeros({max_seq, c.kv_heads, c.head_dim}));
    }
  }

  // Called with the residual stream h at the entry of every layer, and with
  // index n_layers after the final norm (transformers' hidden-state indices).
  std::function<void(std::int64_t, const Tensor&)> hook;

  // Logits for every row when all_logits, else for the last row only.
  void forward(std::span<const std::int32_t> tokens, std::int64_t start_pos, bool all_logits,
               std::vector<float>& logits) {
    const Config& c = w_.cfg;
    const auto n = static_cast<std::int64_t>(tokens.size());
    const std::int64_t qd = c.heads * c.head_dim, kvd = c.kv_heads * c.head_dim;
    Tensor h = Tensor::empty({n, c.hidden});
    if (!v_.embed_q8) {
      ops::embedding(w_.embed.f32, tokens, h);
    } else {
      const std::int64_t nb = c.hidden / kQK;
      for (std::int64_t t = 0; t < n; ++t) {
        const std::int64_t id = tokens[static_cast<std::size_t>(t)];
        if (v_.wq == Weights8::kQ32)
          dequantize_row(w_.embed.q32.data() + id * nb, h.f32() + t * c.hidden, c.hidden);
        else
          dequantize_row((v_.wq == Weights8::kBug ? w_.embed.qbug : w_.embed.q16).data() + id * nb,
                         h.f32() + t * c.hidden, c.hidden);
      }
    }
    std::vector<std::int64_t> pos(static_cast<std::size_t>(n));
    std::iota(pos.begin(), pos.end(), start_pos);
    Tensor x = Tensor::empty({n, c.hidden}), q = Tensor::empty({n, qd}), k = Tensor::empty({n, kvd}),
           v = Tensor::empty({n, kvd}), a = Tensor::empty({n, c.heads, c.head_dim}),
           o = Tensor::empty({n, c.hidden}), g = Tensor::empty({n, c.inter}),
           u = Tensor::empty({n, c.inter}), d = Tensor::empty({n, c.hidden});
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
    const std::int64_t len = start_pos + n;
    for (std::size_t li = 0; li < w_.layers.size(); ++li) {
      const Layer& L = w_.layers[li];
      if (hook) hook(static_cast<std::int64_t>(li), h);
      ops::rmsnorm(h, L.in_norm, c.eps, x);
      linear(x, L.q, v_.linear, v_.wq, q);
      linear(x, L.k, v_.linear, v_.wq, k);
      linear(x, L.v, v_.linear, v_.wq, v);
      Tensor q3 = q.reshape({n, c.heads, c.head_dim});
      Tensor k3 = k.reshape({n, c.kv_heads, c.head_dim});
      ops::rope(q3, &k3, pos, c.theta);
      std::memcpy(kc_[li].f32() + start_pos * kvd, k.f32(), sizeof(float) * n * kvd);
      std::memcpy(vc_[li].f32() + start_pos * kvd, v.f32(), sizeof(float) * n * kvd);
      ops::attention(q3, kc_[li].slice(0, 0, len), vc_[li].slice(0, 0, len), start_pos, scale, a);
      linear(a.reshape({n, qd}), L.o, v_.linear, v_.wq, o);
      ops::add(h, o, h);
      ops::rmsnorm(h, L.post_norm, c.eps, x);
      linear(x, L.gate, v_.linear, v_.wq, g);
      linear(x, L.up, v_.linear, v_.wq, u);
      ops::silu_mul(g, u, g);
      linear(g, L.down, v_.linear, v_.wq, d);
      ops::add(h, d, h);
    }
    ops::rmsnorm(h, w_.final_norm, c.eps, h);
    if (hook) hook(c.layers, h);
    const Tensor rows = all_logits ? h : h.slice(0, n - 1, 1);
    logits.resize(static_cast<std::size_t>(rows.dim(0) * c.vocab));
    Tensor lt = Tensor::wrap(logits.data(), {rows.dim(0), c.vocab});
    linear(rows, w_.embed, v_.head, v_.wq, lt);
  }

  std::vector<std::int32_t> greedy(std::span<const std::int32_t> prompt, int n_gen) {
    std::vector<float> logits;
    std::vector<std::int32_t> out;
    forward(prompt, 0, false, logits);
    std::int64_t pos = static_cast<std::int64_t>(prompt.size());
    for (int i = 0; i < n_gen; ++i) {
      const std::int32_t t = ops::argmax(Tensor::wrap(logits.data(), {w_.cfg.vocab}));
      out.push_back(t);
      if (i + 1 < n_gen) forward(std::span(&t, 1), pos++, false, logits);
    }
    return out;
  }

 private:
  const ModelWeights& w_;
  Variant v_;
  std::vector<Tensor> kc_, vc_;
};

std::int32_t argmax_row(const float* p, std::int64_t n) {
  return static_cast<std::int32_t>(std::max_element(p, p + n) - p);
}

// Top-1 minus top-2 logit: how close a position is to a tie.
double margin(const float* p, std::int64_t n) {
  float a = -std::numeric_limits<float>::infinity(), b = a;
  for (std::int64_t j = 0; j < n; ++j) {
    if (p[j] > a) {
      b = a;
      a = p[j];
    } else if (p[j] > b) {
      b = p[j];
    }
  }
  return static_cast<double>(a) - static_cast<double>(b);
}

// Accumulated comparison of a variant's logits against fp32 logits at the same
// positions.
struct Agreement {
  std::int64_t positions = 0, top1 = 0, nll_count = 0;
  double kl_sum = 0, kl_max = 0, dlogit_max = 0, nll_ref = 0, nll_q = 0;
  double max_flipped_margin = 0;  // largest fp32 top-1 minus top-2 logit at a position that flipped
  std::vector<double> kls;

  // targets[t] is the token that actually follows position t, or -1.
  void add(const std::vector<float>& ref, const std::vector<float>& q, std::int64_t vocab,
           const std::vector<std::int32_t>& targets) {
    const auto n = static_cast<std::int64_t>(ref.size()) / vocab;
    for (std::int64_t t = 0; t < n; ++t) {
      const float* r = ref.data() + t * vocab;
      const float* s = q.data() + t * vocab;
      ++positions;
      if (argmax_row(r, vocab) == argmax_row(s, vocab)) {
        ++top1;
      } else {
        max_flipped_margin = std::max(max_flipped_margin, margin(r, vocab));
      }
      const double mr = *std::max_element(r, r + vocab), ms = *std::max_element(s, s + vocab);
      double zr = 0, zs = 0;
      for (std::int64_t j = 0; j < vocab; ++j) {
        zr += std::exp(r[j] - mr);
        zs += std::exp(s[j] - ms);
        dlogit_max = std::max(dlogit_max, static_cast<double>(std::fabs(r[j] - s[j])));
      }
      const double lzr = std::log(zr) + mr, lzs = std::log(zs) + ms;
      double kl = 0;
      for (std::int64_t j = 0; j < vocab; ++j) {
        const double lpr = r[j] - lzr, lps = s[j] - lzs;
        kl += std::exp(lpr) * (lpr - lps);
      }
      kl = std::max(kl, 0.0);
      kl_sum += kl;
      kl_max = std::max(kl_max, kl);
      kls.push_back(kl);
      const std::int32_t tgt = targets[static_cast<std::size_t>(t)];
      if (tgt >= 0) {
        ++nll_count;
        nll_ref += lzr - r[tgt];
        nll_q += lzs - s[tgt];
      }
    }
  }
  double kl_quantile(double qq) const {
    std::vector<double> v = kls;
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[static_cast<std::size_t>(qq * static_cast<double>(v.size() - 1))];
  }
};

std::vector<std::int32_t> to_i32(const std::vector<std::int64_t>& v) {
  return std::vector<std::int32_t>(v.begin(), v.end());
}

// Natural-text sequences: one per line, "name id id id ...".
std::vector<std::pair<std::string, std::vector<std::int32_t>>> load_eval_ids(const fs::path& p) {
  std::vector<std::pair<std::string, std::vector<std::int32_t>>> out;
  std::ifstream f(p);
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    std::string name;
    ss >> name;
    std::vector<std::int32_t> ids;
    for (std::int32_t id; ss >> id;) ids.push_back(id);
    if (!ids.empty()) out.emplace_back(name, std::move(ids));
  }
  return out;
}

// full = false runs only section E (the kernels against their references).
void quality(ModelWeights& w, const fs::path& oracle, const fs::path& eval_ids,
             const fs::path& torch_dir, bool full) {
  std::printf("\n== quality ==\n");
  const Config& c = w.cfg;
  const int n_prompts = 3, n_gen = 64;

  struct Prompt {
    std::vector<std::int32_t> ids, greedy, seq, targets;
    std::vector<float> oracle_logits;
  };
  std::vector<Prompt> prompts(n_prompts);
  for (int p = 0; p < n_prompts; ++p) {
    const std::string s = "model/p" + std::to_string(p) + "_";
    Prompt& P = prompts[static_cast<std::size_t>(p)];
    P.ids = to_i32(test::read_npy(oracle / (s + "input_ids.npy")).as_i64());
    P.greedy = to_i32(test::read_npy(oracle / (s + "greedy_ids.npy")).as_i64());
    P.oracle_logits = test::read_npy(oracle / (s + "logits.npy")).as_f32();
    // Teacher-forced sequence: the prompt plus the oracle's own continuation.
    P.seq = P.ids;
    P.seq.insert(P.seq.end(), P.greedy.begin(), P.greedy.end());
    P.targets.assign(P.seq.begin() + 1, P.seq.end());
    P.targets.push_back(-1);
  }
  const auto texts = load_eval_ids(eval_ids);
  std::int64_t text_tokens = 0, max_len = 0;
  for (const auto& [name, ids] : texts) {
    text_tokens += static_cast<std::int64_t>(ids.size());
    max_len = std::max<std::int64_t>(max_len, static_cast<std::int64_t>(ids.size()));
  }
  for (const Prompt& P : prompts) max_len = std::max<std::int64_t>(max_len, static_cast<std::int64_t>(P.seq.size()));
  std::printf("eval text: %zu sequences, %lld tokens (%s)\n", texts.size(),
              static_cast<long long>(text_tokens), eval_ids.string().c_str());

  // 1. The harness itself: the fp32 forward pass must reproduce the oracle.
  Runner ref(w, Variant{"f32"}, max_len);
  std::vector<std::vector<float>> ref_prompt_logits(n_prompts), ref_text_logits(texts.size());
  if (full) {
    std::vector<float> lg;
    for (int p = 0; p < n_prompts; ++p) {
      Prompt& P = prompts[static_cast<std::size_t>(p)];
      ref.forward(P.ids, 0, true, lg);
      ErrStats s;
      for (std::size_t i = 0; i < lg.size(); ++i) s.add(lg[i], P.oracle_logits[i], test::tol::kLogits);
      const std::vector<std::int32_t> g = ref.greedy(P.ids, n_gen);
      int same = 0;
      while (same < n_gen && g[static_cast<std::size_t>(same)] == P.greedy[static_cast<std::size_t>(same)]) ++same;
      std::printf("fp32 harness, prompt %d: logits vs oracle max abs %.2e (%s), greedy %d/64 ids match\n",
                  p, s.max_abs, s.within ? "within kLogits" : "OUTSIDE kLogits", same);
      ref.forward(P.seq, 0, true, ref_prompt_logits[static_cast<std::size_t>(p)]);
    }
    for (std::size_t i = 0; i < texts.size(); ++i) ref.forward(texts[i].second, 0, true, ref_text_logits[i]);
  }

  const std::vector<Variant> variants = {
      {"dequant: 7 projections", Kern::kDequant, Kern::kF32, false},
      {"dequant: + lm_head (2 copies)", Kern::kDequant, Kern::kDequant, false},
      {"dequant: all, 1 copy", Kern::kDequant, Kern::kDequant, true},
      {"dequant: all, 1 copy, 36B", Kern::kDequant, Kern::kDequant, true, Weights8::kQ32},
      {"w8a8: 7 projections", Kern::kW8A8, Kern::kF32, false},
      {"w8a8: + lm_head (2 copies)", Kern::kW8A8, Kern::kW8A8, false},
      {"w8a8: all, 1 copy", Kern::kW8A8, Kern::kW8A8, true},
      {"w8a8: all, 1 copy, 36B", Kern::kW8A8, Kern::kW8A8, true, Weights8::kQ32},
      {"BUG trunc: dequant all", Kern::kDequant, Kern::kDequant, true, Weights8::kBug},
  };

  struct Result {
    std::string name;
    Agreement prompts, prose, code;
    std::vector<int> prefix;
  };
  std::vector<Result> results;
  for (const Variant& v : full ? variants : std::vector<Variant>{}) {
    const double t0 = now_s();
    Runner r(w, v, max_len);
    Result res;
    res.name = v.name;
    std::vector<float> lg;
    for (int p = 0; p < n_prompts; ++p) {
      const Prompt& P = prompts[static_cast<std::size_t>(p)];
      r.forward(P.seq, 0, true, lg);
      res.prompts.add(ref_prompt_logits[static_cast<std::size_t>(p)], lg, c.vocab, P.targets);
    }
    for (std::size_t i = 0; i < texts.size(); ++i) {
      const auto& [name, ids] = texts[i];
      std::vector<std::int32_t> tg(ids.begin() + 1, ids.end());
      tg.push_back(-1);
      r.forward(ids, 0, true, lg);
      const bool prose = name.find(".txt:") != std::string::npos;
      (prose ? res.prose : res.code).add(ref_text_logits[i], lg, c.vocab, tg);
    }
    for (int p = 0; p < n_prompts; ++p) {
      const Prompt& P = prompts[static_cast<std::size_t>(p)];
      const std::vector<std::int32_t> g = r.greedy(P.ids, n_gen);
      int same = 0;
      while (same < n_gen && g[static_cast<std::size_t>(same)] == P.greedy[static_cast<std::size_t>(same)]) ++same;
      res.prefix.push_back(same);
    }
    std::printf("  ran %-30s in %.1f s\n", v.name.c_str(), now_s() - t0);
    std::fflush(stdout);
    results.push_back(std::move(res));
  }

  auto pct = [](const Agreement& a) {
    return a.positions ? 100.0 * static_cast<double>(a.top1) / static_cast<double>(a.positions) : 0.0;
  };
  auto kl_mean = [](const Agreement& a) {
    return a.positions ? a.kl_sum / static_cast<double>(a.positions) : 0.0;
  };
  auto ppl = [](double nll, std::int64_t n) { return n ? std::exp(nll / static_cast<double>(n)) : 0.0; };

  if (full) {
  std::printf("\nA. oracle prompts + their 64-token continuations, teacher-forced: %lld positions\n",
              static_cast<long long>(results.empty() ? 0 : results[0].prompts.positions));
  std::printf("  %-30s %7s %10s %10s %10s %10s %11s\n", "variant", "top-1", "KL mean", "KL p99", "KL max",
              "max dlogit", "flip margin");
  for (const Result& r : results)
    std::printf("  %-30s %6.1f%% %10.2e %10.2e %10.2e %10.3f %11.3f\n", r.name.c_str(), pct(r.prompts),
                kl_mean(r.prompts), r.prompts.kl_quantile(0.99), r.prompts.kl_max, r.prompts.dlogit_max,
                r.prompts.max_flipped_margin);

  for (int set = 0; set < 2; ++set) {
    const char* title = set == 0 ? "B. prose (eval_text/*.txt), teacher-forced"
                                 : "C. code (src/ops/matmul.cpp), teacher-forced";
    std::printf("\n%s: %lld positions\n", title,
                static_cast<long long>(results.empty() ? 0
                                       : set == 0     ? results[0].prose.positions
                                                      : results[0].code.positions));
    std::printf("  %-30s %7s %10s %10s %10s %10s %11s %9s %9s %8s\n", "variant", "top-1", "KL mean",
                "KL p99", "KL max", "max dlogit", "flip margin", "PPL f32", "PPL q8", "dPPL");
    for (const Result& r : results) {
      const Agreement& a = set == 0 ? r.prose : r.code;
      const double pr = ppl(a.nll_ref, a.nll_count), pq = ppl(a.nll_q, a.nll_count);
      std::printf("  %-30s %6.1f%% %10.2e %10.2e %10.2e %10.3f %11.3f %9.4f %9.4f %+7.2f%%\n",
                  r.name.c_str(), pct(a), kl_mean(a), a.kl_quantile(0.99), a.kl_max, a.dlogit_max,
                  a.max_flipped_margin, pr, pq, 100.0 * (pq / pr - 1.0));
    }
  }

  std::printf("\nD. free-running greedy (KV cache, one token per step) vs the oracle's 64 ids\n");
  std::printf("  %-30s %s\n", "variant",
              "tokens identical before the first divergence, per prompt [fp32 top-1 minus top-2 logit there]");
  for (const Result& r : results) {
    int sum = 0;
    std::string line;
    for (int p = 0; p < n_prompts; ++p) {
      const int same = r.prefix[static_cast<std::size_t>(p)];
      sum += same;
      char buf[64];
      if (same < n_gen) {
        // Row n_prompt - 1 + same of the teacher-forced fp32 logits predicts greedy token `same`.
        const auto row = static_cast<std::int64_t>(prompts[static_cast<std::size_t>(p)].ids.size()) - 1 + same;
        std::snprintf(buf, sizeof(buf), "p%d %2d [%.3f]  ", p, same,
                      margin(ref_prompt_logits[static_cast<std::size_t>(p)].data() + row * c.vocab, c.vocab));
      } else {
        std::snprintf(buf, sizeof(buf), "p%d %2d          ", p, same);
      }
      line += buf;
    }
    std::printf("  %-30s %s (%d / 192)\n", r.name.c_str(), line.c_str(), sum);
  }
  }  // if (full)

  // E. The Q8_0 kernels against the Q8_0 path's references: the fp32 engine on
  // dequantized weights (C++), and transformers on the same weights (torch).
  std::printf("\nE. Q8_0 kernels vs the Q8_0 references (all quantized, 1 copy, 34B blocks)\n");
  struct Pair {
    Variant kernel, reference;
    const char* torch_mode;
  };
  const Pair pairs[] = {
      {{"dequant kernel", Kern::kDequant, Kern::kDequant, true},
       {"ref: ops::matmul on dequantize(W)", Kern::kRefDequant, Kern::kRefDequant, true}, "dequant"},
      {{"w8a8 kernel", Kern::kW8A8, Kern::kW8A8, true},
       {"ref: ops::matmul on dq(q(x)), dq(W)", Kern::kRefW8A8, Kern::kRefW8A8, true}, "w8a8"},
  };
  for (const Pair& pr : pairs) {
    Runner kr(w, pr.kernel, max_len), rr(w, pr.reference, max_len);
    ErrStats e;
    std::int64_t positions = 0, flips = 0;
    std::vector<float> a, b;
    auto compare = [&](std::span<const std::int32_t> ids) {
      kr.forward(ids, 0, true, a);
      rr.forward(ids, 0, true, b);
      for (std::size_t i = 0; i < a.size(); ++i) e.add(a[i], b[i], test::tol::kLogits);
      for (std::size_t t = 0; t < a.size() / static_cast<std::size_t>(c.vocab); ++t) {
        ++positions;
        if (argmax_row(a.data() + t * c.vocab, c.vocab) != argmax_row(b.data() + t * c.vocab, c.vocab)) ++flips;
      }
    };
    for (const Prompt& P : prompts) compare(P.seq);
    for (const auto& [name, ids] : texts) compare(ids);
    std::string greedy;
    for (const Prompt& P : prompts) {
      const auto g1 = kr.greedy(P.ids, n_gen), g2 = rr.greedy(P.ids, n_gen);
      int same = 0;
      while (same < n_gen && g1[static_cast<std::size_t>(same)] == g2[static_cast<std::size_t>(same)]) ++same;
      greedy += std::to_string(same) + "/64 ";
    }
    std::printf("  %-15s vs %-36s: %lld positions, logits max abs %.2e (%s), top-1 flips %lld, greedy %s\n",
                pr.kernel.name.c_str(), pr.reference.name.c_str(), static_cast<long long>(positions),
                e.max_abs, e.within ? "within kLogits" : "OUTSIDE kLogits", static_cast<long long>(flips),
                greedy.c_str());

    {
      // How the two paths drift apart layer by layer on prompt 2's
      // teacher-forced sequence: max |h_kernel - h_reference| at each layer
      // entry, and how many activation blocks quantize differently there.
      std::vector<Tensor> hk, hr;
      kr.hook = [&](std::int64_t, const Tensor& h) { hk.push_back(h.copy_contiguous()); };
      rr.hook = [&](std::int64_t, const Tensor& h) { hr.push_back(h.copy_contiguous()); };
      kr.forward(prompts[2].seq, 0, false, a);
      rr.forward(prompts[2].seq, 0, false, b);
      kr.hook = nullptr;
      rr.hook = nullptr;
      std::string trace;
      for (std::size_t l = 0; l < hk.size(); ++l) {
        double mx = 0;
        for (std::int64_t i = 0; i < hk[l].numel(); ++i)
          mx = std::max(mx, static_cast<double>(std::fabs(hk[l].f32()[i] - hr[l].f32()[i])));
        if (l % 5 == 0 || l + 1 == hk.size()) {
          char buf[48];
          std::snprintf(buf, sizeof(buf), " h%zu %.1e", l, mx);
          trace += buf;
        }
      }
      std::printf("    residual stream max |kernel - ref|, prompt 2 (79 tokens):%s\n", trace.c_str());
    }
    if (torch_dir.empty()) continue;
    for (Runner* run : {&kr, &rr}) {
      ErrStats t;
      std::string tg;
      for (int p = 0; p < n_prompts; ++p) {
        const Prompt& P = prompts[static_cast<std::size_t>(p)];
        const std::string base = std::string(pr.torch_mode) + "_p" + std::to_string(p) + "_";
        const std::vector<float> tl = test::read_npy(torch_dir / (base + "logits.npy")).as_f32();
        const std::vector<std::int32_t> tgr = to_i32(test::read_npy(torch_dir / (base + "greedy_ids.npy")).as_i64());
        run->forward(P.ids, 0, true, a);
        for (std::size_t i = 0; i < a.size(); ++i) t.add(a[i], tl[i], test::tol::kLogits);
        const auto g = run->greedy(P.ids, n_gen);
        int same = 0;
        while (same < n_gen && g[static_cast<std::size_t>(same)] == tgr[static_cast<std::size_t>(same)]) ++same;
        tg += std::to_string(same) + "/64 ";
      }
      std::printf("  %-36s vs torch %-7s oracle: prompt logits max abs %.2e (%s), greedy %s\n",
                  run == &kr ? pr.kernel.name.c_str() : pr.reference.name.c_str(), pr.torch_mode, t.max_abs,
                  t.within ? "within kLogits" : "OUTSIDE kLogits", tg.c_str());
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <model_dir> <oracle_dir> [selftest] [bench] [threads] [quality|refcheck] "
                 "[torchq8=<dir>]\n",
                 argv[0]);
    return 2;
  }
  std::vector<std::string> sections;
  fs::path torch_dir;  // torchq8=<dir>: output of torch_q8_oracle.py, optional
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a.rfind("torchq8=", 0) == 0)
      torch_dir = a.substr(8);
    else
      sections.push_back(a);
  }
  if (sections.empty()) sections = {"selftest", "bench", "threads", "quality"};
  auto want = [&](const char* s) { return std::find(sections.begin(), sections.end(), s) != sections.end(); };
  try {
    const double t0 = now_s();
    ModelWeights w = load_model(argv[1]);
    std::printf("loaded + quantized %s in %.1f s: %zu projections\n", argv[1], now_s() - t0,
                w.projections().size());
    std::size_t f32 = w.embed.bytes_f32(), q16 = w.embed.bytes_q16(), q32 = w.embed.bytes_q32();
    std::size_t pf32 = 0, pq16 = 0, pq32 = 0;
    for (Linear* l : w.projections()) {
      pf32 += l->bytes_f32();
      pq16 += l->bytes_q16();
      pq32 += l->bytes_q32();
    }
    std::printf("bytes: projections f32 %zu q8_0/34B %zu q8_0/36B %zu | embed f32 %zu q8_0/34B %zu q8_0/36B %zu\n",
                pf32, pq16, pq32, f32, q16, q32);
    if (want("selftest")) selftest(w);
    if (want("bench")) bench(w);
    if (want("threads")) threads(w);
    if (want("quality") || want("refcheck"))
      quality(w, argv[2], fs::path(__FILE__).parent_path() / "eval_ids.txt", torch_dir, want("quality"));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
