// BF16-weight matmul and embedding — owned by workflow 20-bf16-kernels.
//
// Until then a BF16 weight is rejected with std::invalid_argument, as it was
// before ops.h accepted it.

#include "bf16.h"

#include <stdexcept>

namespace minillm::ops::detail {

void matmul_bf16(const Tensor&, const Tensor&, Tensor&) {
  throw std::invalid_argument("ops::matmul: bf16 weights are not implemented yet (workflow 20)");
}

void embedding_bf16(const Tensor&, std::span<const std::int32_t>, Tensor&) {
  throw std::invalid_argument(
      "ops::embedding: bf16 tables are not implemented yet (workflow 20)");
}

}  // namespace minillm::ops::detail
