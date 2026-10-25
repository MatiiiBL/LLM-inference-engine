// ModelConfig — owned by workflow 01-loader.
//
// STUB. Everything that parses config.json throws; attention_scale() is real
// because it is pure arithmetic with nothing to parse.

#include <cmath>
#include <stdexcept>

#include "minillm/config.h"

namespace minillm {

float ModelConfig::attention_scale() const {
  if (head_dim <= 0) {
    throw std::invalid_argument("ModelConfig::attention_scale: head_dim is " +
                                std::to_string(head_dim));
  }
  return 1.0f / std::sqrt(static_cast<float>(head_dim));
}

ModelConfig ModelConfig::from_json_file(const std::filesystem::path&) {
  throw std::logic_error("not implemented: ModelConfig::from_json_file (workflow 01-loader)");
}

ModelConfig ModelConfig::from_json_string(std::string_view) {
  throw std::logic_error("not implemented: ModelConfig::from_json_string (workflow 01-loader)");
}

void ModelConfig::validate() const {
  throw std::logic_error("not implemented: ModelConfig::validate (workflow 01-loader)");
}

std::string ModelConfig::to_string() const {
  throw std::logic_error("not implemented: ModelConfig::to_string (workflow 01-loader)");
}

}  // namespace minillm
