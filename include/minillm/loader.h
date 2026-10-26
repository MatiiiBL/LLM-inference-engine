// minillm — loader.h
//
// Reads safetensors checkpoints into named fp32 tensors. Everything is
// converted to F32 at load time: the rest of the engine never sees bf16/f16.
// Q8_0 tensors exist only where a caller replaces loaded F32 weights with
// quantize_q8_0's output (model.h: quantize_weights_q8_0).
//
// Expected tensor names (Hugging Face Llama):
//   model.embed_tokens.weight                      [vocab_size, hidden_size]
//   model.layers.{i}.input_layernorm.weight        [hidden_size]
//   model.layers.{i}.self_attn.q_proj.weight       [n_heads * head_dim, hidden_size]
//   model.layers.{i}.self_attn.k_proj.weight       [n_kv_heads * head_dim, hidden_size]
//   model.layers.{i}.self_attn.v_proj.weight       [n_kv_heads * head_dim, hidden_size]
//   model.layers.{i}.self_attn.o_proj.weight       [hidden_size, n_heads * head_dim]
//   model.layers.{i}.post_attention_layernorm.weight [hidden_size]
//   model.layers.{i}.mlp.gate_proj.weight          [intermediate_size, hidden_size]
//   model.layers.{i}.mlp.up_proj.weight            [intermediate_size, hidden_size]
//   model.layers.{i}.mlp.down_proj.weight          [hidden_size, intermediate_size]
//   model.norm.weight                              [hidden_size]
//   lm_head.weight                                 [vocab_size, hidden_size]
//     — absent when config.tie_word_embeddings is true, in which case the
//       model reuses model.embed_tokens.weight.
//
// THIS HEADER IS FROZEN. See docs/ARCHITECTURE.md.

#ifndef MINILLM_LOADER_H
#define MINILLM_LOADER_H

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "minillm/tensor.h"

namespace minillm {

/// A bag of named tensors, each contiguous and owning its storage. All F32 as
/// loaded; a caller may replace some with other dtypes (Q8_0 today).
class Weights {
 public:
  Weights() = default;

  bool has(std::string_view name) const;
  /// Throws std::runtime_error naming the missing tensor.
  const Tensor& get(std::string_view name) const;
  /// nullptr if absent.
  const Tensor* find(std::string_view name) const;

  /// Takes ownership of a tensor. Throws std::runtime_error on duplicates.
  void insert(std::string name, Tensor tensor);

  /// Replaces an existing tensor, of any dtype or shape, dropping the old
  /// one's storage once no other handle shares it. Throws std::runtime_error
  /// naming the tensor if it is absent.
  void replace(std::string_view name, Tensor tensor);

  /// Names in sorted order.
  std::vector<std::string> names() const;
  std::size_t size() const;
  /// Total bytes of tensor data held (Tensor::nbytes, so Q8_0 counts blocks).
  std::size_t bytes() const;

 private:
  std::map<std::string, Tensor, std::less<>> tensors_;
};

/// Converts a tensor to a newly allocated contiguous F32 tensor. F32 input is
/// copied unchanged. Q8_0 is dequantized: element j of a block is
/// float(qs[j]) * f16_to_f32(d), which is exact in float.
Tensor convert_to_f32(const Tensor& src);

/// Quantizes a contiguous F32 tensor whose last dimension is a multiple of 32
/// into a new Q8_0 tensor of the same shape. For each block of 32 values x:
///
///   amax = max |x|
///   d    = f32_to_f16(amax / 127.0f)          (the stored scale)
///   q    = clamp(round(x * (1.0f / f16_to_f32(d))), -127, 127)
///
/// all in float, with round() rounding halves away from zero. A block whose
/// stored scale is zero (amax / 127 rounds to 0 in f16, as for an all-zero
/// block) has every q = 0. Quantizing against the stored scale bounds the round-trip
/// error by f16_to_f32(d) / 2 per element. The same input gives the same
/// bytes on every platform.
///
/// Throws std::invalid_argument for a non-F32 or non-contiguous input, a last
/// dimension that is not a multiple of 32, a non-finite value, or a block
/// whose scale overflows f16 (amax / 127 rounds to inf).
Tensor quantize_q8_0(const Tensor& src);

/// Loads one .safetensors file, converting every tensor to F32.
/// Throws std::runtime_error on a missing file, a malformed header, or a
/// truncated data section.
Weights load_safetensors(const std::filesystem::path& file);

/// Loads a model directory: `model.safetensors` if present, otherwise every
/// shard listed in `model.safetensors.index.json`. Throws std::runtime_error
/// if neither exists.
Weights load_checkpoint(const std::filesystem::path& model_dir);

/// One entry of a safetensors header, before any conversion. Exposed so the
/// loader tests (and the weights oracle) can check stored dtypes and shapes
/// without materializing the data.
struct TensorInfo {
  std::string name;
  DType dtype = DType::F32;   ///< dtype as STORED in the file
  Shape shape;
  std::size_t data_begin = 0;  ///< byte offset within the data section
  std::size_t data_end = 0;
};

/// Parses only the header of a .safetensors file, in file order.
/// Throws std::runtime_error on a malformed header.
std::vector<TensorInfo> read_safetensors_header(const std::filesystem::path& file);

}  // namespace minillm

#endif  // MINILLM_LOADER_H
