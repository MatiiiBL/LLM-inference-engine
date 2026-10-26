// BF16-weight kernels for ops::matmul and ops::embedding.
//
// Private to src/ops/ — not part of the public interface. ops::matmul and
// ops::embedding call these when their weight argument is BF16; ops.h
// documents the semantics (a BF16 weight is exactly convert_to_f32 of it, and
// activations stay F32). Each function validates its own arguments.

#ifndef MINILLM_SRC_OPS_BF16_H
#define MINILLM_SRC_OPS_BF16_H

#include <cstdint>
#include <span>

#include "minillm/tensor.h"

namespace minillm::ops::detail {

/// ops::matmul with a BF16 w: out = x · convert_to_f32(w)ᵀ.
void matmul_bf16(const Tensor& x, const Tensor& w, Tensor& out);

/// ops::embedding with a BF16 table: out[i, :] = convert_to_f32(table)[ids[i], :].
void embedding_bf16(const Tensor& table, std::span<const std::int32_t> ids, Tensor& out);

}  // namespace minillm::ops::detail

#endif  // MINILLM_SRC_OPS_BF16_H
