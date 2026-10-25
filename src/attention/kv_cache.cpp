// KVCache — owned by workflow 06.
//
// Implemented in Phase 0 rather than stubbed: it is 100 lines of indexing, and
// having it working means the attention tests can exercise a real cache.
//
// Layout (docs/ARCHITECTURE.md §3): K and V are separate owning F32 tensors of
// shape [n_layers, max_seq_len, n_kv_heads, head_dim]. One (layer, position)
// slot is therefore n_kv_heads * head_dim contiguous floats, and one layer's
// prefix [0, length) is a contiguous [length, n_kv_heads, head_dim] view.

#include "minillm/kv_cache.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace minillm {
namespace {

[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument("KVCache: " + message);
}

[[noreturn]] void out_of_range(const std::string& message) {
  throw std::out_of_range("KVCache: " + message);
}

}  // namespace

KVCache::KVCache(std::int64_t n_layers, std::int64_t max_seq_len, std::int64_t n_kv_heads,
                 std::int64_t head_dim)
    : n_layers_(n_layers),
      max_seq_len_(max_seq_len),
      n_kv_heads_(n_kv_heads),
      head_dim_(head_dim) {
  if (n_layers <= 0 || max_seq_len <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
    fail("all extents must be positive, got n_layers=" + std::to_string(n_layers) +
         " max_seq_len=" + std::to_string(max_seq_len) +
         " n_kv_heads=" + std::to_string(n_kv_heads) + " head_dim=" + std::to_string(head_dim));
  }
  const Shape shape{n_layers, max_seq_len, n_kv_heads, head_dim};
  k_ = Tensor::zeros(shape, DType::F32);
  v_ = Tensor::zeros(shape, DType::F32);
}

KVCache KVCache::for_model(const ModelConfig& cfg, std::int64_t max_seq_len) {
  if (max_seq_len <= 0) {
    fail("max_seq_len must be positive, got " + std::to_string(max_seq_len));
  }
  if (max_seq_len > cfg.max_position_embeddings) {
    fail("max_seq_len " + std::to_string(max_seq_len) + " exceeds the model's " +
         std::to_string(cfg.max_position_embeddings) + " position embeddings");
  }
  return KVCache(cfg.num_hidden_layers, max_seq_len, cfg.num_key_value_heads, cfg.head_dim);
}

void KVCache::set_seq_len(std::int64_t len) {
  if (len < 0 || len > max_seq_len_) {
    out_of_range("seq_len " + std::to_string(len) + " outside [0, " +
                 std::to_string(max_seq_len_) + "]");
  }
  seq_len_ = len;
}

void KVCache::reset() { seq_len_ = 0; }

float* KVCache::k_slot(std::int64_t layer, std::int64_t pos) {
  if (layer < 0 || layer >= n_layers_) {
    out_of_range("layer " + std::to_string(layer) + " outside [0, " +
                 std::to_string(n_layers_) + ")");
  }
  if (pos < 0 || pos >= max_seq_len_) {
    out_of_range("position " + std::to_string(pos) + " outside [0, " +
                 std::to_string(max_seq_len_) + ")");
  }
  return k_.f32() + k_.offset(layer, pos, 0, 0);
}

float* KVCache::v_slot(std::int64_t layer, std::int64_t pos) {
  // Same bounds, same layout: reuse the checked k_slot arithmetic.
  const std::int64_t slot = k_slot(layer, pos) - k_.f32();
  return v_.f32() + slot;
}

void KVCache::store(std::int64_t layer, std::int64_t pos, const float* k, const float* v) {
  if (k == nullptr || v == nullptr) fail("store() given a null k or v pointer");
  const std::int64_t n = n_kv_heads_ * head_dim_;
  float* k_dst = k_slot(layer, pos);
  float* v_dst = v_slot(layer, pos);
  std::memcpy(k_dst, k, static_cast<std::size_t>(n) * sizeof(float));
  std::memcpy(v_dst, v, static_cast<std::size_t>(n) * sizeof(float));
}

Tensor KVCache::k(std::int64_t layer, std::int64_t length) const {
  if (layer < 0 || layer >= n_layers_) {
    out_of_range("layer " + std::to_string(layer) + " outside [0, " +
                 std::to_string(n_layers_) + ")");
  }
  if (length < 0 || length > max_seq_len_) {
    out_of_range("length " + std::to_string(length) + " outside [0, " +
                 std::to_string(max_seq_len_) + "]");
  }
  return k_.row(layer).slice(0, 0, length);
}

Tensor KVCache::v(std::int64_t layer, std::int64_t length) const {
  if (layer < 0 || layer >= n_layers_) {
    out_of_range("layer " + std::to_string(layer) + " outside [0, " +
                 std::to_string(n_layers_) + ")");
  }
  if (length < 0 || length > max_seq_len_) {
    out_of_range("length " + std::to_string(length) + " outside [0, " +
                 std::to_string(max_seq_len_) + "]");
  }
  return v_.row(layer).slice(0, 0, length);
}

std::size_t KVCache::bytes() const {
  return static_cast<std::size_t>(k_.numel() + v_.numel()) * sizeof(float);
}

}  // namespace minillm
