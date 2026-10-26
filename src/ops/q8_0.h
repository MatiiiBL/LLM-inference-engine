// Q8_0 kernels for ops::matmul and ops::embedding.
//
// Private to src/ops/ — not part of the public interface. ops::matmul and
// ops::embedding call these when their weight argument is Q8_0; ops.h
// documents the semantics (a Q8_0 weight is exactly convert_to_f32 of it, and
// activations stay F32). Each function validates its own arguments.

#ifndef MINILLM_SRC_OPS_Q8_0_H
#define MINILLM_SRC_OPS_Q8_0_H

#include <cstdint>
#include <span>

#include "minillm/tensor.h"

namespace minillm::ops::detail {

/// ops::matmul with a Q8_0 w: out = x · convert_to_f32(w)ᵀ.
void matmul_q8_0(const Tensor& x, const Tensor& w, Tensor& out);

/// ops::embedding with a Q8_0 table: out[i, :] = convert_to_f32(table)[ids[i], :].
void embedding_q8_0(const Tensor& table, std::span<const std::int32_t> ids, Tensor& out);

}  // namespace minillm::ops::detail

#endif  // MINILLM_SRC_OPS_Q8_0_H
