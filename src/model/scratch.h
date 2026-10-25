// Model scratch — the buffers one forward pass works in. Private to src/model/.
//
// Every buffer a pass needs comes from one Arena that the Model sizes once,
// in its constructor, for the worst case: a prefill of max_seq_len tokens.
// forward() rewinds the arena and carves these buffers again for its own n,
// so a pass never touches the heap and nothing is ever resized.
//
// scratch_bytes() and carve() must describe the same buffers in the same
// order. tests/model/test_model.cpp holds them to it: an arena of exactly
// scratch_bytes(cfg, n) bytes must fit carve(cfg, n) with Arena::peak() equal
// to its capacity, not a byte to spare and not a byte short.
//
// Header-only so the test can include it without a build-file change.

#ifndef MINILLM_SRC_MODEL_SCRATCH_H
#define MINILLM_SRC_MODEL_SCRATCH_H

#include <cstddef>
#include <cstdint>

#include "minillm/config.h"
#include "minillm/runtime.h"
#include "minillm/tensor.h"

namespace minillm::model_detail {

/// Alignment the arena gives every buffer below: allocate_array and tensor()
/// both ask for 64 bytes, and the arena's block is page-aligned, so the
/// padding before each buffer depends only on the sizes before it.
inline constexpr std::size_t kScratchAlign = 64;

/// The per-pass buffers for n tokens. All are views into the arena, valid
/// until its next reset().
struct Scratch {
  std::int64_t* positions;  ///< [n] absolute positions start_pos + t
  Tensor h;                 ///< [n, hidden] the residual stream
  Tensor x;                 ///< [n, hidden] a block's normed input, then its output
  Tensor q;                 ///< [n, n_heads, head_dim] queries, RoPE'd in place
  Tensor attn;              ///< [n, n_heads, head_dim] attention output
  Tensor gate;              ///< [n, intermediate] gate_proj, then silu(gate) * up
  Tensor up;                ///< [n, intermediate] up_proj
  Tensor logits;            ///< [1, vocab] the last token's logits
};

/// Bytes an arena needs to carve() a Scratch for n tokens, padding included.
inline std::size_t scratch_bytes(const ModelConfig& cfg, std::int64_t n) {
  const auto floats = [](std::int64_t count) { return static_cast<std::size_t>(count) * sizeof(float); };
  const std::size_t sizes[] = {
      static_cast<std::size_t>(n) * sizeof(std::int64_t),  // positions
      floats(n * cfg.hidden_size),                         // h
      floats(n * cfg.hidden_size),                         // x
      floats(n * cfg.q_dim()),                             // q
      floats(n * cfg.q_dim()),                             // attn
      floats(n * cfg.intermediate_size),                   // gate
      floats(n * cfg.intermediate_size),                   // up
      floats(cfg.vocab_size),                              // logits
  };
  std::size_t used = 0;
  for (const std::size_t size : sizes) {
    used = (used + kScratchAlign - 1) / kScratchAlign * kScratchAlign + size;
  }
  return used;
}

/// Carves the buffers for n tokens out of `arena`, in scratch_bytes() order.
inline Scratch carve(Arena& arena, const ModelConfig& cfg, std::int64_t n) {
  Scratch s;
  s.positions = arena.allocate_array<std::int64_t>(static_cast<std::size_t>(n));
  s.h = arena.tensor(Shape{n, cfg.hidden_size});
  s.x = arena.tensor(Shape{n, cfg.hidden_size});
  s.q = arena.tensor(Shape{n, cfg.num_attention_heads, cfg.head_dim});
  s.attn = arena.tensor(Shape{n, cfg.num_attention_heads, cfg.head_dim});
  s.gate = arena.tensor(Shape{n, cfg.intermediate_size});
  s.up = arena.tensor(Shape{n, cfg.intermediate_size});
  s.logits = arena.tensor(Shape{1, cfg.vocab_size});
  return s;
}

}  // namespace minillm::model_detail

#endif  // MINILLM_SRC_MODEL_SCRATCH_H
