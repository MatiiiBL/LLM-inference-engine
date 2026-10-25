// Weights + safetensors loading — owned by workflow 01-loader.
//
// STUB, except for the Weights container itself: it is a map with error
// messages, there is nothing to discover about it, and having it working means
// other modules can be tested with hand-built weights before the real loader
// exists.

#include <stdexcept>

#include "minillm/loader.h"

namespace minillm {

bool Weights::has(std::string_view name) const { return tensors_.find(name) != tensors_.end(); }

const Tensor* Weights::find(std::string_view name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const Tensor& Weights::get(std::string_view name) const {
  const Tensor* t = find(name);
  if (t == nullptr) {
    throw std::runtime_error("Weights: no tensor named '" + std::string(name) + "' (" +
                             std::to_string(tensors_.size()) + " tensors loaded)");
  }
  return *t;
}

void Weights::insert(std::string name, Tensor tensor) {
  const auto [it, inserted] = tensors_.emplace(std::move(name), std::move(tensor));
  if (!inserted) {
    throw std::runtime_error("Weights: duplicate tensor name '" + it->first + "'");
  }
}

std::vector<std::string> Weights::names() const {
  std::vector<std::string> out;
  out.reserve(tensors_.size());
  for (const auto& [name, tensor] : tensors_) out.push_back(name);
  return out;  // std::map iterates in sorted order
}

std::size_t Weights::size() const { return tensors_.size(); }

std::size_t Weights::bytes() const {
  std::size_t total = 0;
  for (const auto& [name, tensor] : tensors_) {
    total += static_cast<std::size_t>(tensor.numel()) * dtype_size(tensor.dtype());
  }
  return total;
}

Tensor convert_to_f32(const Tensor&) {
  throw std::logic_error("not implemented: convert_to_f32 (workflow 01-loader)");
}

Weights load_safetensors(const std::filesystem::path&) {
  throw std::logic_error("not implemented: load_safetensors (workflow 01-loader)");
}

Weights load_checkpoint(const std::filesystem::path&) {
  throw std::logic_error("not implemented: load_checkpoint (workflow 01-loader)");
}

std::vector<TensorInfo> read_safetensors_header(const std::filesystem::path&) {
  throw std::logic_error("not implemented: read_safetensors_header (workflow 01-loader)");
}

}  // namespace minillm
