// Q8_0 matmul and embedding — owned by workflow 16-q8-kernels.

#include "q8_0.h"

#include <stdexcept>

namespace minillm::ops::detail {

void matmul_q8_0(const Tensor&, const Tensor&, Tensor&) {
  throw std::logic_error("not implemented: q8_0 ops::matmul (workflow 16)");
}

void embedding_q8_0(const Tensor&, std::span<const std::int32_t>, Tensor&) {
  throw std::logic_error("not implemented: q8_0 ops::embedding (workflow 16)");
}

}  // namespace minillm::ops::detail
