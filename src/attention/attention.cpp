// ops::attention — owned by workflow 06.
//
// Causal grouped-query attention in one pass over the keys, with an online
// softmax (the flash-attention recurrence, on a CPU):
//
//   * Keys are visited in blocks of kKeyBlock. A head's scores for one block
//     live on the stack; its running max and running sum are updated once
//     per block, and the output accumulated so far is rescaled only when the
//     max grows. The kernel therefore never allocates and has no length
//     limit.
//   * The query heads that share a KV head (one GQA group) are processed as
//     tiles of up to kMaxTile heads, and every K or V vector loaded is used by
//     every head of the tile. Loads, not multiplies, are what bound this
//     kernel, so a tile of three runs close to three times the heads per
//     cycle of one head at a time.
//   * The inner loops keep their working set in registers: the tile's query
//     slices while scoring a block of keys, its output slices while summing
//     the block's V rows. Accumulating the output in memory instead makes
//     every key wait on the previous key's stores.
//   * Arithmetic is spelled out on 4-float vectors (f32x4), the softmax's
//     max, exp and sum included. Left to itself the auto-vectorizer builds
//     the dot products out of 2-wide operations and cannot vectorize the
//     reductions at all without -ffast-math, which this file does not need:
//     every reassociation is written out explicitly.
//   * Threads split the (query row, KV head) pairs. Each pair's output rows
//     are computed start to finish by one thread, exactly as on one thread,
//     so the result does not depend on the thread count.
//
// Causality is the loop bound, exactly as in the reference: query row t reads
// keys [0, start_pos + t] and nothing past them.
//
// The signature takes K/V as tensor views rather than a KVCache on purpose
// (see docs/ARCHITECTURE.md §3): it keeps the reference independent of the
// cache implementation and lets tests feed K/V straight from .npy files.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

#include "minillm/ops.h"
#include "minillm/runtime.h"

namespace minillm::ops {
namespace {

/// Smallest share of a call worth handing to another thread, in multiply-adds.
constexpr std::int64_t kMinChunkMacs = std::int64_t{1} << 13;

/// Keys per block. One block's scores for a tile sit on the stack.
constexpr std::int64_t kKeyBlock = 128;
/// Query heads per tile. Three heads' kSlice-float query or output slices
/// are 24 vectors, which leaves room for the loads in NEON's 32 registers.
/// SmolLM2's GQA group is exactly 3; a larger group takes several tiles.
constexpr std::int64_t kMaxTile = 3;
/// Floats of head_dim handled per pass over a block. head_dim % kSlice
/// trailing dimensions go through a scalar loop.
constexpr std::int64_t kSlice = 32;

/// Four floats in one SIMD register: NEON on arm64, SSE on x86-64. GCC and
/// Clang, the supported compilers, lower arithmetic on it directly to vector
/// instructions. Comparisons yield an i32x4 lane mask usable with ?:.
using f32x4 = float __attribute__((vector_size(16)));
using i32x4 = std::int32_t __attribute__((vector_size(16)));

f32x4 load4(const float* p) {
  f32x4 v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

void store4(float* p, f32x4 v) { std::memcpy(p, &v, sizeof(v)); }

f32x4 splat4(float x) { return f32x4{x, x, x, x}; }

float sum4(f32x4 v) { return (v[0] + v[2]) + (v[1] + v[3]); }

/// e^x for x <= 0, four lanes at a time — the softmax only ever exponentiates
/// scores minus their running max. Cody-Waite reduction x = n ln2 + r with
/// |r| <= ln2 / 2, the Cephes degree-5 polynomial for e^r, and 2^n built
/// directly in the exponent bits.
///
/// Measured over every float in [-87.34, 0]: at most 1.01 ulp from the
/// correctly rounded result (libm's expf: 0.51 ulp). Below -87.34, where e^x
/// drops under the smallest normal float (1.2e-38), the result is exactly 0
/// rather than a denormal; next to the block's largest weight, which is 1,
/// that is invisible. NaN propagates and -inf gives 0.
f32x4 exp4(f32x4 x) {
  constexpr float kMin = -87.33654f;      // e^kMin = 2^-126
  constexpr float kLog2e = 1.44269504088896341f;
  constexpr float kLn2Hi = 0.693359375f;  // few mantissa bits: n * kLn2Hi is exact
  constexpr float kLn2Lo = -2.12194440e-4f;
  constexpr float kRound = 12582912.0f;   // 1.5 * 2^23: adding it rounds to an integer

  const i32x4 underflow = x < kMin;
  x = underflow ? splat4(kMin) : x;
  const f32x4 t = x * kLog2e + kRound;
  const f32x4 n = t - kRound;
  const f32x4 r = (x - n * kLn2Hi) - n * kLn2Lo;

  f32x4 p = 1.9875691500e-4f * r + 1.3981999507e-3f;
  p = p * r + 8.3334519073e-3f;
  p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f;
  p = p * r + 5.0000001201e-1f;
  const f32x4 er = (p * (r * r) + r) + 1.0f;

  // t's low mantissa bits hold n exactly; rebias it as a float exponent.
  const i32x4 n_bits = std::bit_cast<i32x4>(t) - std::bit_cast<i32x4>(splat4(kRound));
  const f32x4 two_n = std::bit_cast<f32x4>((n_bits + 127) << 23);
  return underflow ? splat4(0.0f) : er * two_n;
}

[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument("ops::attention: " + message);
}

void require_f32_rank3(const char* name, const Tensor& t) {
  if (t.dtype() != DType::F32) fail(std::string(name) + " must be f32, got " + t.to_string());
  if (t.ndim() != 3) fail(std::string(name) + " must have rank 3, got " + t.to_string());
}

/// The reference's checks, so a bad call throws std::invalid_argument
/// whichever implementation it reaches.
void validate(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
              std::int64_t start_pos, const Tensor& out) {
  require_f32_rank3("q", q);
  if (!q.is_contiguous()) fail("q must be contiguous, got " + q.to_string());
  require_f32_rank3("out", out);
  if (!out.is_contiguous()) fail("out must be contiguous, got " + out.to_string());
  // The cache views may be strided (a slice of the full cache).
  require_f32_rank3("k_cache", k_cache);
  require_f32_rank3("v_cache", v_cache);
  if (q.shape() != out.shape()) {
    fail("q is " + q.shape().to_string() + " but out is " + out.shape().to_string());
  }
  if (k_cache.shape() != v_cache.shape()) {
    fail("k_cache is " + k_cache.shape().to_string() + " but v_cache is " +
         v_cache.shape().to_string());
  }

  const std::int64_t n_tokens = q.shape()[0];
  const std::int64_t n_q_heads = q.shape()[1];
  const std::int64_t head_dim = q.shape()[2];
  const std::int64_t n_kv = k_cache.shape()[0];
  const std::int64_t n_kv_heads = k_cache.shape()[1];

  if (k_cache.shape()[2] != head_dim) {
    fail("k_cache head_dim is " + std::to_string(k_cache.shape()[2]) + ", expected " +
         std::to_string(head_dim) + " (q head_dim)");
  }
  if (start_pos < 0) fail("start_pos must be >= 0, got " + std::to_string(start_pos));
  if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0) {
    fail("n_q_heads " + std::to_string(n_q_heads) + " is not a multiple of n_kv_heads " +
         std::to_string(n_kv_heads));
  }
  // Written as a subtraction so a huge start_pos cannot overflow.
  if (start_pos > n_kv - n_tokens) {
    fail("cache holds " + std::to_string(n_kv) + " positions but the pass needs " +
         std::to_string(start_pos) + " + " + std::to_string(n_tokens));
  }
}

/// True if each [head_dim] row of a cache view is packed. Every view the
/// Tensor API can cut from a contiguous buffer qualifies, whatever its outer
/// strides; the kernel indexes rows through stride(0) and stride(1).
bool rows_are_packed(const Tensor& t) { return t.shape()[2] <= 1 || t.stride(2) == 1; }

/// One block's scores for a tile: s[h][j] for head h of the tile, key j.
template <std::int64_t kHeads>
using Scores = float[kHeads][kKeyBlock];

/// s[h][j] += dot(q_h[0:kSlice], k_j[0:kSlice]) for each head h of the tile
/// and each of n keys. q_h is h * head_dim floats past q and k_j is
/// j * k_stride floats past k. The tile's query slices stay in registers,
/// and each K vector is loaded once for all kHeads heads.
template <std::int64_t kHeads>
void add_slice_scores(const float* q, std::int64_t head_dim, const float* k,
                      std::int64_t k_stride, std::int64_t n, Scores<kHeads>& s) {
  constexpr std::int64_t kVecs = kSlice / 4;
  f32x4 qv[kHeads][kVecs];
  for (std::int64_t h = 0; h < kHeads; ++h) {
    for (std::int64_t i = 0; i < kVecs; ++i) qv[h][i] = load4(q + h * head_dim + 4 * i);
  }

  for (std::int64_t j = 0; j < n; ++j) {
    const float* kj = k + j * k_stride;
    f32x4 acc[kHeads] = {};
    for (std::int64_t i = 0; i < kVecs; ++i) {
      const f32x4 kv = load4(kj + 4 * i);
      for (std::int64_t h = 0; h < kHeads; ++h) acc[h] += qv[h][i] * kv;
    }
    for (std::int64_t h = 0; h < kHeads; ++h) s[h][j] += sum4(acc[h]);
  }
}

/// out_h[0:kSlice] += sum over j of p[h][j] * v_j[0:kSlice] for each head h
/// of the tile, out_h being h * head_dim floats past out. The tile's output
/// slices stay in registers, and each V vector is loaded once for all heads.
template <std::int64_t kHeads>
void add_slice_weighted_sum(const Scores<kHeads>& p, const float* v, std::int64_t v_stride,
                            std::int64_t n, float* out, std::int64_t head_dim) {
  constexpr std::int64_t kVecs = kSlice / 4;
  f32x4 acc[kHeads][kVecs];
  for (std::int64_t h = 0; h < kHeads; ++h) {
    for (std::int64_t i = 0; i < kVecs; ++i) acc[h][i] = load4(out + h * head_dim + 4 * i);
  }

  for (std::int64_t j = 0; j < n; ++j) {
    const float* vj = v + j * v_stride;
    f32x4 pj[kHeads];
    for (std::int64_t h = 0; h < kHeads; ++h) pj[h] = splat4(p[h][j]);
    for (std::int64_t i = 0; i < kVecs; ++i) {
      const f32x4 vv = load4(vj + 4 * i);
      for (std::int64_t h = 0; h < kHeads; ++h) acc[h][i] += pj[h] * vv;
    }
  }

  for (std::int64_t h = 0; h < kHeads; ++h) {
    for (std::int64_t i = 0; i < kVecs; ++i) store4(out + h * head_dim + 4 * i, acc[h][i]);
  }
}

/// Turns one block of raw dot products s[0, n) into softmax weights relative
/// to the head's running max, and folds them into the running sum. If the
/// block raises the max, the head's partial output, accumulated against the
/// old max, is rescaled first.
void softmax_block(float* s, std::int64_t n, float scale, bool first_block, float& running_max,
                   float& running_sum, float* out, std::int64_t head_dim) {
  // Scale, and take the block max four lanes at a time. max is exact, so the
  // lane split cannot change the result.
  std::int64_t j = 0;
  f32x4 max4 = splat4(-std::numeric_limits<float>::infinity());
  for (; j + 4 <= n; j += 4) {
    const f32x4 x = load4(s + j) * scale;
    store4(s + j, x);
    max4 = x > max4 ? x : max4;
  }
  float block_max = std::max(std::max(max4[0], max4[1]), std::max(max4[2], max4[3]));
  for (; j < n; ++j) {
    s[j] *= scale;
    block_max = std::max(block_max, s[j]);
  }

  if (block_max > running_max) {
    if (!first_block) {  // before the first block there is nothing to rescale
      const float correction = std::exp(running_max - block_max);
      running_sum *= correction;
      for (std::int64_t d = 0; d < head_dim; ++d) out[d] *= correction;
    }
    running_max = block_max;
  }

  // Weights, four lanes at a time. The last partial vector is padded with
  // -inf, whose weight is exactly 0.
  const float m = running_max;
  f32x4 total = splat4(0.0f);
  for (j = 0; j + 4 <= n; j += 4) {
    const f32x4 w = exp4(load4(s + j) - m);
    store4(s + j, w);
    total += w;
  }
  if (j < n) {
    f32x4 x = splat4(-std::numeric_limits<float>::infinity());
    for (std::int64_t l = 0; j + l < n; ++l) x[l] = s[j + l] - m;
    const f32x4 w = exp4(x);
    for (std::int64_t l = 0; j + l < n; ++l) s[j + l] = w[l];
    total += w;
  }
  running_sum += sum4(total);
}

/// One query token against keys [0, n_keys) of one KV head, for a tile of
/// kHeads consecutive query heads of that head's group.
///
///   q, out  kHeads rows of head_dim floats, contiguous; out is overwritten
///   k, v    that KV head's row at key 0; key j is j * k_stride (v_stride)
///           floats further on
template <std::int64_t kHeads>
void attend(const float* q, float* out, const float* k, std::int64_t k_stride, const float* v,
            std::int64_t v_stride, std::int64_t n_keys, std::int64_t head_dim, float scale) {
  Scores<kHeads> s;
  float running_max[kHeads];
  float running_sum[kHeads];
  for (std::int64_t h = 0; h < kHeads; ++h) {
    running_max[h] = -std::numeric_limits<float>::infinity();
    running_sum[h] = 0.0f;
  }
  std::fill_n(out, kHeads * head_dim, 0.0f);
  const std::int64_t sliced = head_dim - head_dim % kSlice;

  for (std::int64_t j0 = 0; j0 < n_keys; j0 += kKeyBlock) {
    const std::int64_t n_block = std::min(kKeyBlock, n_keys - j0);
    const float* k_block = k + j0 * k_stride;
    const float* v_block = v + j0 * v_stride;

    for (std::int64_t h = 0; h < kHeads; ++h) std::fill_n(s[h], n_block, 0.0f);
    for (std::int64_t d = 0; d < sliced; d += kSlice) {
      add_slice_scores<kHeads>(q + d, head_dim, k_block + d, k_stride, n_block, s);
    }
    for (std::int64_t d = sliced; d < head_dim; ++d) {
      for (std::int64_t h = 0; h < kHeads; ++h) {
        for (std::int64_t j = 0; j < n_block; ++j) {
          s[h][j] += q[h * head_dim + d] * k_block[j * k_stride + d];
        }
      }
    }

    for (std::int64_t h = 0; h < kHeads; ++h) {
      softmax_block(s[h], n_block, scale, j0 == 0, running_max[h], running_sum[h],
                    out + h * head_dim, head_dim);
    }

    // Unnormalized; divided by the running sum once all blocks are in.
    for (std::int64_t d = 0; d < sliced; d += kSlice) {
      add_slice_weighted_sum<kHeads>(s, v_block + d, v_stride, n_block, out + d, head_dim);
    }
    for (std::int64_t d = sliced; d < head_dim; ++d) {
      for (std::int64_t h = 0; h < kHeads; ++h) {
        float acc = out[h * head_dim + d];
        for (std::int64_t j = 0; j < n_block; ++j) acc += s[h][j] * v_block[j * v_stride + d];
        out[h * head_dim + d] = acc;
      }
    }
  }

  for (std::int64_t h = 0; h < kHeads; ++h) {
    float* oh = out + h * head_dim;
    for (std::int64_t d = 0; d < head_dim; ++d) oh[d] /= running_sum[h];
  }
}

}  // namespace

void attention(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
               std::int64_t start_pos, float scale, Tensor& out) {
  validate(q, k_cache, v_cache, start_pos, out);
  if (!rows_are_packed(k_cache) || !rows_are_packed(v_cache)) {
    // Not reachable through the Tensor API today; read it correctly, slowly,
    // rather than wrongly.
    ref::attention(q, k_cache, v_cache, start_pos, scale, out);
    return;
  }

  const std::int64_t n_tokens = q.shape()[0];
  const std::int64_t n_q_heads = q.shape()[1];
  const std::int64_t head_dim = q.shape()[2];
  const std::int64_t n_kv_heads = k_cache.shape()[1];
  const std::int64_t group = n_q_heads / n_kv_heads;
  const std::int64_t ks = k_cache.stride(0);
  const std::int64_t vs = v_cache.stride(0);

  const float* pq = q.f32();
  const float* pk = k_cache.f32();
  const float* pv = v_cache.f32();
  float* po = out.f32();

  // Unit u is one (query row, KV head) pair: every query head of that KV
  // head's group, for that row. Rows are numbered from the last, so the
  // pairs with the most keys are claimed first and the cheap ones even out
  // the end.
  const auto run = [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t u = begin; u < end; ++u) {
      const std::int64_t t = n_tokens - 1 - u / n_kv_heads;
      const std::int64_t g = u % n_kv_heads;
      // Query row t sits at absolute position start_pos + t and may look at key
      // positions 0 .. start_pos + t inclusive.
      const std::int64_t n_keys = start_pos + t + 1;

      // Query heads g * group .. (g + 1) * group - 1 read KV head g.
      const float* k = pk + g * k_cache.stride(1);
      const float* v = pv + g * v_cache.stride(1);
      const std::int64_t h_end = (g + 1) * group;
      for (std::int64_t h0 = g * group; h0 < h_end;) {
        const std::int64_t tile = std::min(kMaxTile, h_end - h0);
        const float* qt = pq + (t * n_q_heads + h0) * head_dim;
        float* ot = po + (t * n_q_heads + h0) * head_dim;
        static_assert(kMaxTile == 3, "instantiate attend<> for every tile size");
        switch (tile) {
          case 3: attend<3>(qt, ot, k, ks, v, vs, n_keys, head_dim, scale); break;
          case 2: attend<2>(qt, ot, k, ks, v, vs, n_keys, head_dim, scale); break;
          default: attend<1>(qt, ot, k, ks, v, vs, n_keys, head_dim, scale); break;
        }
        h0 += tile;
      }
    }
  };
  // Multiply-adds per unit, for a row of average length: q·k and p·v.
  const std::int64_t unit_macs =
      std::max<std::int64_t>(1, 2 * group * head_dim * (start_pos + (n_tokens + 1) / 2));
  kernel_parallel_for(n_tokens * n_kv_heads, (kMinChunkMacs + unit_macs - 1) / unit_macs,
                      unit_macs * n_tokens * n_kv_heads, std::ref(run));
}

}  // namespace minillm::ops
