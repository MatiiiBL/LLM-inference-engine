// ops::rmsnorm, ops::softmax — owned by workflow 03.
//
// Phase 0: forwards to the reference. See src/ops/elementwise.cpp for the
// rules on replacing these.

#include "minillm/ops.h"

namespace minillm::ops {

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) {
  ref::rmsnorm(x, weight, eps, out);
}

void softmax(Tensor& x) { ref::softmax(x); }

}  // namespace minillm::ops
