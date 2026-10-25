// Arena — owned by workflow 08-runtime.
//
// STUB. Construction records the requested capacity; every allocation throws.

#include <stdexcept>

#include "minillm/runtime.h"

namespace minillm {

struct Arena::Impl {
  std::size_t capacity = 0;
};

Arena::Arena(std::size_t bytes) : impl_(std::make_unique<Impl>()) { impl_->capacity = bytes; }

Arena::~Arena() = default;

void* Arena::allocate(std::size_t, std::size_t) {
  throw std::logic_error("not implemented: Arena::allocate (workflow 08-runtime)");
}

Tensor Arena::tensor(const Shape&, DType) {
  throw std::logic_error("not implemented: Arena::tensor (workflow 08-runtime)");
}

void Arena::reset() {
  throw std::logic_error("not implemented: Arena::reset (workflow 08-runtime)");
}

std::size_t Arena::used() const { return 0; }

std::size_t Arena::capacity() const { return impl_->capacity; }

std::size_t Arena::peak() const { return 0; }

}  // namespace minillm
