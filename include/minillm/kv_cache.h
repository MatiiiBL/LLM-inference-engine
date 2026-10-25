// minillm — kv_cache.h
//
// Preallocated key/value cache. One contiguous buffer for K and one for V,
// both laid out [layer][position][kv_head][head_dim], allocated once for
// max_seq_len positions and never resized or reallocated during generation.
//
// THIS HEADER IS FROZEN. See docs/ARCHITECTURE.md.

#ifndef MINILLM_KV_CACHE_H
#define MINILLM_KV_CACHE_H

#include <cstddef>
#include <cstdint>

#include "minillm/config.h"
#include "minillm/tensor.h"

namespace minillm {

/// Causal KV cache for one sequence (no batching).
///
/// Layout: K and V are separate owning F32 tensors of shape
/// [n_layers, max_seq_len, n_kv_heads, head_dim], row-major. Position p of
/// layer l is therefore one contiguous run of n_kv_heads * head_dim floats.
class KVCache {
 public:
  /// Empty cache: all extents 0, no allocation.
  KVCache() = default;

  /// Allocates (and zeroes) both buffers. Throws std::invalid_argument if any
  /// extent is <= 0.
  KVCache(std::int64_t n_layers, std::int64_t max_seq_len, std::int64_t n_kv_heads,
          std::int64_t head_dim);

  /// Convenience constructor: sizes from the model config, capped at
  /// max_seq_len positions. Throws std::invalid_argument if max_seq_len is
  /// <= 0 or greater than cfg.max_position_embeddings.
  static KVCache for_model(const ModelConfig& cfg, std::int64_t max_seq_len);

  std::int64_t n_layers() const { return n_layers_; }
  std::int64_t max_seq_len() const { return max_seq_len_; }
  std::int64_t n_kv_heads() const { return n_kv_heads_; }
  std::int64_t head_dim() const { return head_dim_; }

  /// Number of positions written so far, i.e. the next free position.
  std::int64_t seq_len() const { return seq_len_; }
  /// Declares that positions [0, len) hold valid keys and values. The model
  /// calls this once per forward pass, after writing. Throws
  /// std::out_of_range if len is outside [0, max_seq_len].
  void set_seq_len(std::int64_t len);
  /// Forgets all cached positions (seq_len() becomes 0). Does not free or
  /// clear memory; stale values are never read because reads are bounded by
  /// the length passed to k()/v().
  void reset();

  /// Copies one token's keys and values into (layer, pos).
  ///
  ///   k, v  n_kv_heads * head_dim contiguous floats, [kv_head][head_dim]
  ///
  /// Throws std::out_of_range if layer or pos is out of range. Does NOT
  /// advance seq_len() — call set_seq_len() when the pass is done.
  void store(std::int64_t layer, std::int64_t pos, const float* k, const float* v);

  /// Writable pointer to (layer, pos): n_kv_heads * head_dim contiguous
  /// floats. For callers that project straight into the cache instead of
  /// going through store(). Throws std::out_of_range.
  float* k_slot(std::int64_t layer, std::int64_t pos);
  float* v_slot(std::int64_t layer, std::int64_t pos);

  /// Non-owning view [length, n_kv_heads, head_dim] of one layer's cache,
  /// ready to pass to ops::attention. Contiguous. Throws std::out_of_range if
  /// layer is out of range or length is outside [0, max_seq_len].
  Tensor k(std::int64_t layer, std::int64_t length) const;
  Tensor v(std::int64_t layer, std::int64_t length) const;

  /// Total bytes held by both buffers.
  std::size_t bytes() const;

 private:
  Tensor k_;  ///< [n_layers, max_seq_len, n_kv_heads, head_dim] F32
  Tensor v_;  ///< same shape as k_
  std::int64_t n_layers_ = 0;
  std::int64_t max_seq_len_ = 0;
  std::int64_t n_kv_heads_ = 0;
  std::int64_t head_dim_ = 0;
  std::int64_t seq_len_ = 0;
};

}  // namespace minillm

#endif  // MINILLM_KV_CACHE_H
