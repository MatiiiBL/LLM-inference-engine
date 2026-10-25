// Model — the forward pass. Assembled after Wave 1, once the loader, the ops
// and the KV cache all work.
//
// STUB. The data flow this must implement is written out step by step in
// docs/ARCHITECTURE.md §4, including the exact hidden-state hook indices that
// make oracle/data/model/*_hidden_{i}.npy comparable element for element.

#include <stdexcept>

#include "minillm/model.h"

namespace minillm {

struct Model::Impl {
  ModelConfig config;
  Weights weights;
  KVCache cache;
  Model::HiddenStateHook hook;
};

Model::Model(ModelConfig, Weights, std::int64_t) {
  throw std::logic_error("not implemented: Model::Model (assembled after Wave 1)");
}

Model::~Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

Model Model::from_dir(const std::filesystem::path&, std::int64_t) {
  throw std::logic_error("not implemented: Model::from_dir (assembled after Wave 1)");
}

std::span<const float> Model::forward(std::span<const std::int32_t>, std::int64_t) {
  throw std::logic_error("not implemented: Model::forward (assembled after Wave 1)");
}

const ModelConfig& Model::config() const {
  throw std::logic_error("not implemented: Model::config (assembled after Wave 1)");
}

KVCache& Model::kv_cache() {
  throw std::logic_error("not implemented: Model::kv_cache (assembled after Wave 1)");
}

const KVCache& Model::kv_cache() const {
  throw std::logic_error("not implemented: Model::kv_cache (assembled after Wave 1)");
}

void Model::reset() {
  throw std::logic_error("not implemented: Model::reset (assembled after Wave 1)");
}

void Model::set_hidden_state_hook(HiddenStateHook) {
  throw std::logic_error("not implemented: Model::set_hidden_state_hook (assembled after Wave 1)");
}

}  // namespace minillm
