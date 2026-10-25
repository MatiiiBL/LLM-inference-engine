// ModelConfig — owned by workflow 01-loader.
//
// Every hyperparameter the forward pass depends on numerically is REQUIRED in
// the JSON: a missing rope_theta must not quietly become 10000. Keys with a
// documented Hugging Face default (num_key_value_heads, the bias flags,
// hidden_act, ...) fall back to that default when absent.

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "minillm/config.h"

namespace minillm {
namespace {

using json = nlohmann::json;

/// Reads a JSON config from `source` (a file path, or a label for in-memory
/// text); every error message starts with it.
class ConfigReader {
 public:
  ConfigReader(const json& root, std::string source) : root_(root), source_(std::move(source)) {}

  [[noreturn]] void fail(const std::string& message) const {
    throw std::runtime_error("ModelConfig: " + source_ + ": " + message);
  }

  /// The value for `key`, or nullptr if the key is absent or null.
  const json* find(const char* key) const {
    const auto it = root_.find(key);
    return (it == root_.end() || it->is_null()) ? nullptr : &*it;
  }

  const json& require(const char* key) const {
    const json* v = find(key);
    if (v == nullptr) fail(std::string("missing required key \"") + key + "\"");
    return *v;
  }

  std::int64_t as_int64(const char* key, const json& v) const {
    if (!v.is_number_integer()) fail(std::string("\"") + key + "\" must be an integer, got " + v.dump());
    if (v.is_number_unsigned() &&
        v.get<std::uint64_t>() > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      fail(std::string("\"") + key + "\" is out of range: " + v.dump());
    }
    return v.get<std::int64_t>();
  }

  std::int64_t int64(const char* key) const { return as_int64(key, require(key)); }

  float number(const char* key) const {
    const json& v = require(key);
    if (!v.is_number()) fail(std::string("\"") + key + "\" must be a number, got " + v.dump());
    const double value = v.get<double>();
    // Casting an out-of-range double to float is undefined behaviour.
    if (std::abs(value) > std::numeric_limits<float>::max()) {
      fail(std::string("\"") + key + "\" is out of range for float: " + v.dump());
    }
    return static_cast<float>(value);
  }

  bool boolean(const char* key, bool fallback) const {
    const json* v = find(key);
    if (v == nullptr) return fallback;
    if (!v->is_boolean()) fail(std::string("\"") + key + "\" must be true or false, got " + v->dump());
    return v->get<bool>();
  }

  std::string string(const char* key, const std::string& fallback) const {
    const json* v = find(key);
    if (v == nullptr) return fallback;
    if (!v->is_string()) fail(std::string("\"") + key + "\" must be a string, got " + v->dump());
    return v->get<std::string>();
  }

  /// A special-token id: an integer, null/absent (-1), or a list of integers,
  /// of which the first is taken (several configs list multiple EOS ids).
  std::int32_t token_id(const char* key) const {
    const json* v = find(key);
    if (v == nullptr) return -1;
    const json* id = v;
    if (v->is_array()) {
      if (v->empty()) return -1;
      id = &v->front();
    }
    const std::int64_t value = as_int64(key, *id);
    if (value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max()) {
      fail(std::string("\"") + key + "\" does not fit in int32: " + std::to_string(value));
    }
    return static_cast<std::int32_t>(value);
  }

 private:
  const json& root_;
  std::string source_;
};

ModelConfig parse(std::string_view text, const std::string& source) {
  json root;
  try {
    root = json::parse(text);
  } catch (const json::exception& e) {
    throw std::runtime_error("ModelConfig: " + source + ": not valid JSON: " + e.what());
  }
  const ConfigReader r(root, source);
  if (!root.is_object()) r.fail("top level must be a JSON object");

  // No field records rope_scaling, so validate() cannot see it: reject here.
  if (const json* scaling = r.find("rope_scaling")) {
    r.fail("\"rope_scaling\" is " + scaling->dump() + "; RoPE scaling is not implemented");
  }

  ModelConfig cfg;
  cfg.vocab_size = r.int64("vocab_size");
  cfg.hidden_size = r.int64("hidden_size");
  cfg.intermediate_size = r.int64("intermediate_size");
  cfg.num_hidden_layers = r.int64("num_hidden_layers");
  cfg.num_attention_heads = r.int64("num_attention_heads");
  // Absent means plain multi-head attention, exactly as in transformers.
  const json* kv_heads = r.find("num_key_value_heads");
  cfg.num_key_value_heads =
      kv_heads ? r.as_int64("num_key_value_heads", *kv_heads) : cfg.num_attention_heads;
  if (const json* head_dim = r.find("head_dim")) {
    cfg.head_dim = r.as_int64("head_dim", *head_dim);
  } else if (cfg.num_attention_heads > 0) {
    if (cfg.hidden_size % cfg.num_attention_heads != 0) {
      r.fail("no \"head_dim\" and hidden_size " + std::to_string(cfg.hidden_size) +
             " is not divisible by num_attention_heads " + std::to_string(cfg.num_attention_heads));
    }
    cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
  }  // else head_dim stays 0 and validate() reports the head count
  cfg.max_position_embeddings = r.int64("max_position_embeddings");
  cfg.rms_norm_eps = r.number("rms_norm_eps");
  cfg.rope_theta = r.number("rope_theta");
  cfg.tie_word_embeddings = r.boolean("tie_word_embeddings", false);
  cfg.attention_bias = r.boolean("attention_bias", false);
  cfg.mlp_bias = r.boolean("mlp_bias", false);
  cfg.hidden_act = r.string("hidden_act", "silu");
  cfg.model_type = r.string("model_type", "");
  cfg.bos_token_id = r.token_id("bos_token_id");
  cfg.eos_token_id = r.token_id("eos_token_id");

  try {
    cfg.validate();
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(source + ": " + e.what());
  }
  return cfg;
}

}  // namespace

float ModelConfig::attention_scale() const {
  if (head_dim <= 0) {
    throw std::invalid_argument("ModelConfig::attention_scale: head_dim is " +
                                std::to_string(head_dim));
  }
  return 1.0f / std::sqrt(static_cast<float>(head_dim));
}

ModelConfig ModelConfig::from_json_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("ModelConfig: cannot open " + path.string());
  std::ostringstream text;
  text << in.rdbuf();
  if (in.bad()) throw std::runtime_error("ModelConfig: error reading " + path.string());
  return parse(text.str(), path.string());
}

ModelConfig ModelConfig::from_json_string(std::string_view json) {
  return parse(json, "<config.json string>");
}

void ModelConfig::validate() const {
  const auto fail = [](const std::string& message) {
    throw std::runtime_error("ModelConfig::validate: " + message);
  };
  const auto require_positive = [&](const char* name, std::int64_t value) {
    if (value <= 0) fail(std::string(name) + " must be positive, got " + std::to_string(value));
  };
  require_positive("vocab_size", vocab_size);
  require_positive("hidden_size", hidden_size);
  require_positive("intermediate_size", intermediate_size);
  require_positive("num_hidden_layers", num_hidden_layers);
  require_positive("num_attention_heads", num_attention_heads);
  require_positive("num_key_value_heads", num_key_value_heads);
  require_positive("head_dim", head_dim);
  require_positive("max_position_embeddings", max_position_embeddings);

  if (num_key_value_heads > num_attention_heads) {
    fail("num_key_value_heads (" + std::to_string(num_key_value_heads) +
         ") exceeds num_attention_heads (" + std::to_string(num_attention_heads) + ")");
  }
  if (num_attention_heads % num_key_value_heads != 0) {
    fail("num_attention_heads (" + std::to_string(num_attention_heads) +
         ") is not divisible by num_key_value_heads (" + std::to_string(num_key_value_heads) + ")");
  }
  // Not in the header's list, but a zero, negative or NaN eps/theta computes
  // garbage just as surely as a zero dimension does.
  if (!(std::isfinite(rms_norm_eps) && rms_norm_eps > 0.0f)) {
    fail("rms_norm_eps must be positive and finite, got " + std::to_string(rms_norm_eps));
  }
  if (!(std::isfinite(rope_theta) && rope_theta > 0.0f)) {
    fail("rope_theta must be positive and finite, got " + std::to_string(rope_theta));
  }
  if (hidden_act != "silu") {
    fail("hidden_act is \"" + hidden_act + "\"; only \"silu\" (SwiGLU) is implemented");
  }
  if (attention_bias) fail("attention_bias is set; biased attention projections are not implemented");
  if (mlp_bias) fail("mlp_bias is set; biased MLP projections are not implemented");
}

std::string ModelConfig::to_string() const {
  std::ostringstream os;
  os << "ModelConfig (" << (model_type.empty() ? "unknown model_type" : model_type) << ")\n"
     << "  vocab_size              " << vocab_size << '\n'
     << "  hidden_size             " << hidden_size << '\n'
     << "  intermediate_size       " << intermediate_size << '\n'
     << "  num_hidden_layers       " << num_hidden_layers << '\n'
     << "  num_attention_heads     " << num_attention_heads << '\n'
     << "  num_key_value_heads     " << num_key_value_heads << '\n'
     << "  head_dim                " << head_dim << '\n'
     << "  max_position_embeddings " << max_position_embeddings << '\n'
     << "  rms_norm_eps            " << rms_norm_eps << '\n'
     << "  rope_theta              " << rope_theta << '\n'
     << "  hidden_act              " << hidden_act << '\n'
     << "  tie_word_embeddings     " << (tie_word_embeddings ? "true" : "false") << '\n'
     << "  attention_bias          " << (attention_bias ? "true" : "false") << '\n'
     << "  mlp_bias                " << (mlp_bias ? "true" : "false") << '\n'
     << "  bos_token_id            " << bos_token_id << '\n'
     << "  eos_token_id            " << eos_token_id;
  return os.str();
}

}  // namespace minillm
