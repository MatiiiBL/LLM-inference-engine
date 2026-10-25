// ops::rope — owned by workflow 05.
//
// Phase 0: forwards to the reference, which recomputes cos/sin per element.
// An optimized version may cache tables keyed on (head_dim, theta) as long as
// the values are bit-identical to recomputing them, and must keep the
// rotate_half pairing (i with i + head_dim/2) documented in ops.h.

#include "minillm/ops.h"

namespace minillm::ops {

void rope(Tensor& q, Tensor* k, std::span<const std::int64_t> positions, float theta) {
  ref::rope(q, k, positions, theta);
}

}  // namespace minillm::ops
