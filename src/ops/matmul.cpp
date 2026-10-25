// ops::matmul — owned by workflow 04.
//
// Phase 0: forwards to the reference. This is the hot kernel of the engine
// (every linear layer), so it is the one most worth optimizing — but only
// after the tests comparing it to minillm::ref and to oracle/data/ops/ pass.

#include "minillm/ops.h"

namespace minillm::ops {

void matmul(const Tensor& x, const Tensor& w, Tensor& out) { ref::matmul(x, w, out); }

}  // namespace minillm::ops
