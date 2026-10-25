// ops::embedding, ops::silu_mul, ops::add, ops::argmax — owned by workflow 03.
//
// Phase 0: every function forwards to the reference. Replacing a body here is
// the whole job: keep the signature and the documented semantics in ops.h,
// then prove equivalence against minillm::ref on random inputs AND against
// the PyTorch golden data in oracle/data/ops/.

#include "minillm/ops.h"

namespace minillm::ops {

void embedding(const Tensor& table, std::span<const std::int32_t> ids, Tensor& out) {
  ref::embedding(table, ids, out);
}

void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out) {
  ref::silu_mul(gate, up, out);
}

void add(const Tensor& a, const Tensor& b, Tensor& out) { ref::add(a, b, out); }

std::int32_t argmax(const Tensor& x) { return ref::argmax(x); }

}  // namespace minillm::ops
