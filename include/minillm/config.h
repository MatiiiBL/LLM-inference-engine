// minillm — config.h
//
// The model hyperparameters, parsed from a Hugging Face `config.json`.
// NOTHING here is hardcoded for a particular checkpoint: every field is read
// from the file, and validate() rejects configurations this engine does not
// implement rather than silently computing the wrong thing.
//
// THIS HEADER IS FROZEN. See docs/ARCHITECTURE.md.

#ifndef MINILLM_CONFIG_H
#define MINILLM_CONFIG_H

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace minillm {

/// Llama-family model hyperparameters.
///
/// Field names mirror the JSON keys so the mapping is obvious. Values are the
/// defaults for "nothing was specified", not for any particular model — a
/// freshly default-constructed ModelConfig is invalid and validate() says so.
struct ModelConfig {
  // ---- Read directly from config.json -------------------------------------
  std::int64_t vocab_size = 0;               ///< "vocab_size"
  std::int64_t hidden_size = 0;              ///< "hidden_size" (d_model)
  std::int64_t intermediate_size = 0;        ///< "intermediate_size" (MLP width)
  std::int64_t num_hidden_layers = 0;        ///< "num_hidden_layers"
  std::int64_t num_attention_heads = 0;      ///< "num_attention_heads"
  std::int64_t num_key_value_heads = 0;      ///< "num_key_value_heads" (GQA)
  /// "head_dim" if present, else hidden_size / num_attention_heads.
  std::int64_t head_dim = 0;
  std::int64_t max_position_embeddings = 0;  ///< "max_position_embeddings"
  float rms_norm_eps = 0.0f;                 ///< "rms_norm_eps"
  float rope_theta = 0.0f;                   ///< "rope_theta" (RoPE base)
  bool tie_word_embeddings = false;          ///< "tie_word_embeddings"
  bool attention_bias = false;               ///< "attention_bias"
  bool mlp_bias = false;                     ///< "mlp_bias"
  std::string hidden_act = "silu";           ///< "hidden_act"
  std::string model_type;                    ///< "model_type"
  std::int32_t bos_token_id = -1;            ///< "bos_token_id", -1 if absent
  std::int32_t eos_token_id = -1;            ///< "eos_token_id", -1 if absent

  // ---- Derived -------------------------------------------------------------
  /// Queries per KV head. Query head h reads KV head h / kv_group_size().
  std::int64_t kv_group_size() const { return num_attention_heads / num_key_value_heads; }
  /// Width of the fused Q projection output: num_attention_heads * head_dim.
  std::int64_t q_dim() const { return num_attention_heads * head_dim; }
  /// Width of one K or V projection output: num_key_value_heads * head_dim.
  std::int64_t kv_dim() const { return num_key_value_heads * head_dim; }
  /// Softmax scale for attention logits: 1 / sqrt(head_dim).
  float attention_scale() const;

  // ---- Construction --------------------------------------------------------
  /// Parses config.json. Calls validate() before returning.
  /// Throws std::runtime_error if the file cannot be read or parsed, or if it
  /// contains a non-null "rope_scaling" (unimplemented — the only check that
  /// cannot live in validate(), since no field records it).
  static ModelConfig from_json_file(const std::filesystem::path& path);
  /// Parses a config.json document already in memory. Same checks as above.
  static ModelConfig from_json_string(std::string_view json);

  /// Throws std::runtime_error describing the first problem found. Rejects:
  /// non-positive dimensions; num_attention_heads not divisible by
  /// num_key_value_heads; num_key_value_heads > num_attention_heads;
  /// hidden_act != "silu"; attention_bias or mlp_bias set (unimplemented);
  /// max_position_embeddings <= 0.
  void validate() const;

  /// Multi-line human-readable dump, used by the CLI banner and test failures.
  std::string to_string() const;
};

}  // namespace minillm

#endif  // MINILLM_CONFIG_H
