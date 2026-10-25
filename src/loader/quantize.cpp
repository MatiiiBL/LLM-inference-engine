// quantize_q8_0 — owned by workflow 15-q8-quantize.

#include <stdexcept>

#include "minillm/loader.h"

namespace minillm {

Tensor quantize_q8_0(const Tensor&) {
  throw std::logic_error("not implemented: quantize_q8_0 (workflow 15)");
}

}  // namespace minillm
