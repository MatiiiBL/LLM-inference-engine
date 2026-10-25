// Reference implementation: causal grouped-query attention over a KV cache.
//
// FROZEN ORACLE. One query head at a time, scores in a scratch vector, softmax
// over the causal prefix only — there is no mask tensor, the prefix length IS
// the mask.

#include <cmath>
#include <vector>

#include "check.h"
#include "minillm/ops.h"

namespace minillm::ref {

void attention(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
               std::int64_t start_pos, float scale, Tensor& out) {
  detail::require_f32_contiguous_rank("attention", "q", q, 3);
  detail::require_f32_contiguous_rank("attention", "out", out, 3);
  // The cache views may be strided (a slice of the full cache), so only rank
  // and dtype are required here; indexing goes through Tensor::offset().
  detail::require_f32("attention", "k_cache", k_cache);
  detail::require_f32("attention", "v_cache", v_cache);
  detail::require_rank("attention", "k_cache", k_cache, 3);
  detail::require_rank("attention", "v_cache", v_cache, 3);
  detail::require_same_shape("attention", "q", q, "out", out);
  detail::require_same_shape("attention", "k_cache", k_cache, "v_cache", v_cache);

  const std::int64_t n_tokens = q.shape()[0];
  const std::int64_t n_q_heads = q.shape()[1];
  const std::int64_t head_dim = q.shape()[2];
  const std::int64_t n_kv = k_cache.shape()[0];
  const std::int64_t n_kv_heads = k_cache.shape()[1];

  detail::require_dim("attention", "k_cache", k_cache, 2, head_dim, "q head_dim");
  if (start_pos < 0) {
    detail::fail("attention", "start_pos must be >= 0, got " + std::to_string(start_pos));
  }
  if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0) {
    detail::fail("attention", "n_q_heads " + std::to_string(n_q_heads) +
                                  " is not a multiple of n_kv_heads " +
                                  std::to_string(n_kv_heads));
  }
  if (start_pos + n_tokens > n_kv) {
    detail::fail("attention", "cache holds " + std::to_string(n_kv) +
                                  " positions but the pass needs " +
                                  std::to_string(start_pos + n_tokens));
  }

  const std::int64_t kv_group_size = n_q_heads / n_kv_heads;

  const float* pq = q.f32();
  const float* pk = k_cache.f32();
  const float* pv = v_cache.f32();
  float* po = out.f32();

  std::vector<float> scores(static_cast<std::size_t>(n_kv));

  for (std::int64_t t = 0; t < n_tokens; ++t) {
    // Query row t sits at absolute position start_pos + t and may look at key
    // positions 0 .. start_pos + t inclusive.
    const std::int64_t n_keys = start_pos + t + 1;

    for (std::int64_t h = 0; h < n_q_heads; ++h) {
      const std::int64_t h_kv = h / kv_group_size;
      const float* qh = pq + (t * n_q_heads + h) * head_dim;

      for (std::int64_t j = 0; j < n_keys; ++j) {
        const float* kj = pk + k_cache.offset(j, h_kv, 0);
        float dot = 0.0f;
        for (std::int64_t d = 0; d < head_dim; ++d) {
          dot += qh[d] * kj[d];
        }
        scores[static_cast<std::size_t>(j)] = dot * scale;
      }

      // Softmax over [0, n_keys), max-subtracted. Inlined rather than calling
      // ref::softmax so the reference stays a single readable function and
      // does not need a Tensor view over a std::vector.
      float max_v = scores[0];
      for (std::int64_t j = 1; j < n_keys; ++j) {
        if (scores[static_cast<std::size_t>(j)] > max_v) max_v = scores[static_cast<std::size_t>(j)];
      }
      float sum = 0.0f;
      for (std::int64_t j = 0; j < n_keys; ++j) {
        scores[static_cast<std::size_t>(j)] = std::exp(scores[static_cast<std::size_t>(j)] - max_v);
        sum += scores[static_cast<std::size_t>(j)];
      }
      for (std::int64_t j = 0; j < n_keys; ++j) {
        scores[static_cast<std::size_t>(j)] /= sum;
      }

      float* oh = po + (t * n_q_heads + h) * head_dim;
      for (std::int64_t d = 0; d < head_dim; ++d) {
        oh[d] = 0.0f;
      }
      for (std::int64_t j = 0; j < n_keys; ++j) {
        const float weight = scores[static_cast<std::size_t>(j)];
        const float* vj = pv + v_cache.offset(j, h_kv, 0);
        for (std::int64_t d = 0; d < head_dim; ++d) {
          oh[d] += weight * vj[d];
        }
      }
    }
  }
}

}  // namespace minillm::ref
