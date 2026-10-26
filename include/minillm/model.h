// minillm — model.h
//
// The Llama forward pass. One sequence, no batching, fp32 compute.
//
// THIS HEADER IS FROZEN. See docs/ARCHITECTURE.md.

#ifndef MINILLM_MODEL_H
#define MINILLM_MODEL_H

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>

#include "minillm/config.h"
#include "minillm/kv_cache.h"
#include "minillm/loader.h"
#include "minillm/tensor.h"

namespace minillm {

/// Replaces, in place, the seven projections of every layer (q, k, v, o,
/// gate, up, down) and model.embed_tokens.weight with quantize_q8_0 copies;
/// lm_head.weight too when it is present (untied). With tied embeddings the
/// one Q8_0 table serves as both the lookup and lm_head. Norm weights stay
/// F32. Each F32 tensor is freed as soon as its Q8_0 copy replaces it, so
/// peak memory stays at the F32 size. Throws like quantize_q8_0, or
/// std::runtime_error if a required tensor is missing.
void quantize_weights_q8_0(Weights& weights, const ModelConfig& cfg);

/// A loaded model plus its KV cache.
///
/// One forward pass (see docs/ARCHITECTURE.md for the full data flow):
///   h = embed_tokens[tokens]
///   for each layer: h += attn(rmsnorm(h)); h += mlp(rmsnorm(h))
///   h = rmsnorm(h); logits = h · lm_headᵀ   (last token only)
class Model {
 public:
  /// Takes ownership of the weights. `max_seq_len` sizes the KV cache and is
  /// capped by cfg.max_position_embeddings.
  ///
  /// Throws std::runtime_error if a required tensor is missing or has the
  /// wrong shape for `cfg` — all shape checking happens here, so the forward
  /// pass itself can assume consistency.
  ///
  /// The projections and the embedding (and an untied lm_head) may each be
  /// F32 or Q8_0, as quantize_weights_q8_0 leaves them; every other tensor
  /// must be F32. A tensor of any other dtype is a std::runtime_error.
  Model(ModelConfig cfg, Weights weights, std::int64_t max_seq_len);
  ~Model();

  Model(Model&&) noexcept;
  Model& operator=(Model&&) noexcept;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  /// Loads config.json + the safetensors checkpoint from a model directory.
  /// `weight_dtype` F32 (the default) keeps the weights as loaded; Q8_0 runs
  /// quantize_weights_q8_0 on them first. Any other dtype throws
  /// std::invalid_argument before anything is read.
  static Model from_dir(const std::filesystem::path& model_dir, std::int64_t max_seq_len,
                        DType weight_dtype = DType::F32);

  /// Runs `tokens` through the model, appending their keys and values to the
  /// KV cache at positions [start_pos, start_pos + tokens.size()).
  ///
  /// Returns the logits for the LAST token only: a [vocab_size] span into
  /// model-owned scratch, valid until the next forward() call. Prefill and
  /// decode are the same call — prefill passes the whole prompt with
  /// start_pos = 0, decode passes one token with start_pos = cache length.
  ///
  /// Throws std::invalid_argument if tokens is empty, if an id is outside
  /// [0, vocab_size), or if start_pos + tokens.size() exceeds the cache.
  std::span<const float> forward(std::span<const std::int32_t> tokens, std::int64_t start_pos);

  const ModelConfig& config() const;
  KVCache& kv_cache();
  const KVCache& kv_cache() const;

  /// Clears the KV cache so the next forward() can start at position 0.
  void reset();

  /// Observer for hidden states, used to compare against the oracle.
  /// Signature: (index, hidden) where `hidden` is a [n_tokens, hidden_size]
  /// view valid only for the duration of the call.
  ///
  /// Indices match transformers' `output_hidden_states` tuple exactly:
  ///   0 .. num_hidden_layers-1 : the INPUT to decoder layer i
  ///                              (index 0 is the embedding output)
  ///   num_hidden_layers        : hidden states AFTER the final norm
  /// so there are num_hidden_layers + 1 calls per forward pass.
  using HiddenStateHook = std::function<void(std::int64_t index, const Tensor& hidden)>;
  /// Installs (or, with a default-constructed hook, removes) the observer.
  void set_hidden_state_hook(HiddenStateHook hook);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace minillm

#endif  // MINILLM_MODEL_H
