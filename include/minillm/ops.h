// minillm — ops.h
//
// Every kernel the forward pass needs, as free functions. There are two
// namespaces with IDENTICAL signatures:
//
//   minillm::ref — the naive scalar fp32 reference (src/ops_ref/). Obviously
//                  correct, never optimized, NEVER CHANGED. It is the in-repo
//                  oracle: every optimized kernel is tested against it. It
//                  takes F32 only: where an ops:: function also accepts a Q8_0
//                  argument, the reference for that case is the same ref::
//                  function applied to convert_to_f32 of it (loader.h).
//   minillm::ops — what the engine calls. In Phase 0 each function forwards to
//                  minillm::ref; later each one is replaced by a fast version.
//
// Rules that apply to EVERY op in BOTH namespaces:
//   * All tensors are DType::F32 unless stated otherwise. Passing another
//     dtype throws std::invalid_argument.
//   * All inputs and outputs must be contiguous (is_contiguous()). The one
//     exception is the K/V cache views passed to attention(), which are
//     documented individually below.
//   * Output tensors are pre-allocated by the caller with the exact expected
//     shape. Ops never allocate and never resize.
//   * Shapes are validated on entry. A mismatch throws std::invalid_argument
//     with a message naming the op and the offending shapes. This is not a
//     debug-only check: optimized implementations must validate too.
//   * No aliasing between inputs and outputs unless the op says otherwise.
//   * Ops are pure w.r.t. global state and must be safe to call concurrently
//     on disjoint data.
//
// THIS HEADER IS FROZEN. See docs/ARCHITECTURE.md.

#ifndef MINILLM_OPS_H
#define MINILLM_OPS_H

#include <cstdint>
#include <span>

#include "minillm/tensor.h"

namespace minillm::ops {

/// Embedding lookup: out[i, :] = table[ids[i], :]
///
///   table  [vocab_size, hidden_size]  F32 or Q8_0
///   ids    [n_tokens]                 token ids, each in [0, vocab_size)
///   out    [n_tokens, hidden_size]    F32, fully overwritten
///
/// A Q8_0 table's rows are dequantized (each element q * d, computed in
/// float): bit for bit ref::embedding(convert_to_f32(table), ids, out).
///
/// Throws std::out_of_range if any id is outside [0, vocab_size).
void embedding(const Tensor& table, std::span<const std::int32_t> ids, Tensor& out);

/// Matrix multiply against a transposed weight: out = x · Wᵀ
///
///   x    [m, k]  F32
///   w    [n, k]  F32 or Q8_0 — Linear weights are stored [out_features,
///                      in_features], exactly as in the HF checkpoint. Row j
///                      of w is the weight vector for output feature j.
///   out  [m, n]  F32, fully overwritten
///
///   out[i, j] = sum over p in [0, k) of x[i, p] * w[j, p]
///
/// No bias: the target architecture has none (attention_bias = mlp_bias =
/// false) and ModelConfig::validate() rejects models that do.
/// Accumulation is in float. The reference accumulates in source order
/// (p ascending); optimized versions may reassociate, which is why matmul
/// tolerances are looser than elementwise ones (see tests/common/tolerances.h).
///
/// A Q8_0 w stands for exactly the F32 matrix convert_to_f32(w), whose
/// elements are q * d computed in float. The result is that matmul,
/// reassociated: within tol::kMatmul of ref::matmul(x, convert_to_f32(w)).
/// Activations are never quantized.
void matmul(const Tensor& x, const Tensor& w, Tensor& out);

/// Root-mean-square layer norm, row-wise over the last dimension:
///
///   out[i, j] = x[i, j] / sqrt(mean_j(x[i, j]^2) + eps) * weight[j]
///
///   x       [m, k]  F32
///   weight  [k]     F32
///   out     [m, k]  F32. May alias x exactly (same data pointer and shape);
///                   partial overlap is undefined.
///
/// mean and the reciprocal square root are computed in float, matching
/// transformers' LlamaRMSNorm with the input already in float32.
void rmsnorm(const Tensor& x, const Tensor& weight, float eps, Tensor& out);

/// SwiGLU elementwise stage: out = silu(gate) * up, with
///
///   silu(v) = v * sigmoid(v) = v / (1 + exp(-v))
///
///   gate  [m, d]  F32 — the gate_proj output
///   up    [m, d]  F32 — the up_proj output
///   out   [m, d]  F32. May alias gate or up exactly.
void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out);

/// Elementwise add, used for residual connections: out = a + b
///
///   a, b, out — identical shapes, F32. out may alias a or b exactly.
void add(const Tensor& a, const Tensor& b, Tensor& out);

/// Row-wise softmax over the last dimension, IN PLACE:
///
///   x[i, j] <- exp(x[i, j] - max_j x[i, j]) / sum_j exp(x[i, j] - max_j x[i, j])
///
///   x  [m, n] or [n]  F32, modified in place.
///
/// The max subtraction is mandatory (numerical stability) and is part of the
/// contract: implementations must produce the same result for inputs shifted
/// by a constant, up to rounding.
void softmax(Tensor& x);

/// Rotary position embeddings, IN PLACE, Hugging Face `rotate_half` layout.
///
///   q          [n_tokens, n_q_heads, head_dim]   F32, rotated in place
///   k          [n_tokens, n_kv_heads, head_dim]  F32, rotated in place;
///              may be nullptr to rotate q alone (used by tests).
///   positions  [n_tokens]  absolute position of each token, >= 0
///   theta      RoPE base (ModelConfig::rope_theta)
///
/// head_dim must be even. With half = head_dim / 2, for each head:
///
///   inv_freq[i] = theta ^ (-2i / head_dim)        for i in [0, half)
///   angle       = positions[t] * inv_freq[i]
///   x[i]        <-  x[i] * cos(angle) - x[i + half] * sin(angle)
///   x[i + half] <-  x[i + half] * cos(angle) + x[i] * sin(angle)
///
/// i.e. dimension i pairs with dimension i + half — NOT with i + 1. This is
/// the convention baked into HF checkpoints (transformers' rotate_half); the
/// interleaved GPT-NeoX-style pairing would silently produce wrong outputs.
///
/// inv_freq must be computed the way transformers computes it: a float32
/// exponent, a float32 pow, a float32 reciprocal. Computing it in double and
/// narrowing is more accurate in isolation but differs by 1 ulp, and since the
/// angle is position * inv_freq in float32, a 1-ulp relative difference is an
/// absolute angle error of position * 6e-8 radians — 6e-5 at position 4000,
/// straight into the output because cos and sin are 1-Lipschitz. cos/sin
/// themselves are evaluated in double and rounded to float.
/// An implementation may cache tables keyed on (head_dim, theta).
void rope(Tensor& q, Tensor* k, std::span<const std::int64_t> positions, float theta);

/// Causal grouped-query attention over a KV cache.
///
///   q          [n_tokens, n_q_heads, head_dim]   F32, already RoPE'd
///   k_cache    [n_kv, n_kv_heads, head_dim]      F32, already RoPE'd
///   v_cache    [n_kv, n_kv_heads, head_dim]      F32
///   start_pos  absolute position of q row 0
///   scale      attention logit scale, normally 1 / sqrt(head_dim)
///   out        [n_tokens, n_q_heads, head_dim]   F32, fully overwritten
///
/// Query head h reads KV head h / (n_q_heads / n_kv_heads); n_q_heads must be
/// a multiple of n_kv_heads.
///
/// Query row t has absolute position start_pos + t and attends to key
/// positions j in [0, start_pos + t] — strictly causal, no sliding window, no
/// additive mask argument. n_kv must be >= start_pos + n_tokens; keys beyond
/// start_pos + n_tokens - 1 are ignored, so it is fine to pass a cache view
/// that is longer than needed.
///
///   scores[j] = scale * dot(q[t, h, :], k_cache[j, h_kv, :])
///   p         = softmax(scores over j in [0, start_pos + t])
///   out[t, h, :] = sum_j p[j] * v_cache[j, h_kv, :]
///
/// k_cache and v_cache may be non-contiguous views as long as they are
/// slices along dimension 0 of a [max_seq_len, n_kv_heads, head_dim] buffer
/// (which is what KVCache hands out, and which is contiguous anyway). All
/// other tensors must be contiguous.
void attention(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
               std::int64_t start_pos, float scale, Tensor& out);

/// Index of the maximum element of a rank-1 F32 tensor (greedy decoding).
/// Ties resolve to the lowest index, matching torch.argmax. Throws
/// std::invalid_argument if x is not rank 1 or is empty.
std::int32_t argmax(const Tensor& x);

}  // namespace minillm::ops

namespace minillm::ref {

// The reference implementations. Signatures are identical to minillm::ops
// above, and so is the documented behaviour — see there for shapes, aliasing
// rules and formulas — except that every argument must be F32: a Q8_0 one
// throws std::invalid_argument. These are naive scalar loops and are never optimized:
// they are the oracle the fast kernels are compared against.

void embedding(const Tensor& table, std::span<const std::int32_t> ids, Tensor& out);
void matmul(const Tensor& x, const Tensor& w, Tensor& out);
void rmsnorm(const Tensor& x, const Tensor& weight, float eps, Tensor& out);
void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out);
void add(const Tensor& a, const Tensor& b, Tensor& out);
void softmax(Tensor& x);
void rope(Tensor& q, Tensor* k, std::span<const std::int64_t> positions, float theta);
void attention(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
               std::int64_t start_pos, float scale, Tensor& out);
std::int32_t argmax(const Tensor& x);

}  // namespace minillm::ref

#endif  // MINILLM_OPS_H
