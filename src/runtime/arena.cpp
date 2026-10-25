// Arena — owned by workflow 08-runtime.
//
// One block reserved up front and a byte offset into it. allocate() pads the
// offset up to the requested alignment and bumps it; reset() sets it back to
// zero. The block itself is aligned to kBaseAlign, so the padding — and with
// it used() and peak() — depends only on the sequence of requests, never on
// where the system allocator happened to put the block.
//
// Under AddressSanitizer everything not currently handed out is poisoned, so
// reading a view after reset() or running off the end of an allocation is
// reported at the faulting access instead of surfacing later as a wrong
// number somewhere downstream.

#include <algorithm>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "minillm/runtime.h"

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define MINILLM_ARENA_ASAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define MINILLM_ARENA_ASAN 1
#endif

#ifdef MINILLM_ARENA_ASAN
#include <sanitizer/asan_interface.h>
#endif

namespace minillm {
namespace {

/// Alignment of the block. A page: no request in this engine asks for more.
constexpr std::size_t kBaseAlign = 4096;

void poison(void* p, std::size_t bytes) {
#ifdef MINILLM_ARENA_ASAN
  ASAN_POISON_MEMORY_REGION(p, bytes);
#else
  (void)p;
  (void)bytes;
#endif
}

void unpoison(void* p, std::size_t bytes) {
#ifdef MINILLM_ARENA_ASAN
  ASAN_UNPOISON_MEMORY_REGION(p, bytes);
#else
  (void)p;
  (void)bytes;
#endif
}

/// std::bad_alloc that says how far over the arena the request was, which is
/// the number whoever sized the arena needs.
class ArenaExhausted : public std::bad_alloc {
 public:
  explicit ArenaExhausted(std::string what) : what_(std::move(what)) {}
  const char* what() const noexcept override { return what_.c_str(); }

 private:
  std::string what_;
};

}  // namespace

struct Arena::Impl {
  std::byte* base = nullptr;
  std::size_t capacity = 0;
  std::size_t used = 0;
  std::size_t peak = 0;

  ~Impl() {
    unpoison(base, capacity);
    ::operator delete(base, std::align_val_t{kBaseAlign});
  }

  [[noreturn]] void exhausted(const std::string& request) const {
    throw ArenaExhausted("Arena: cannot allocate " + request + ": " +
                         std::to_string(capacity - used) + " of " + std::to_string(capacity) +
                         " bytes free");
  }
};

Arena::Arena(std::size_t bytes) : impl_(std::make_unique<Impl>()) {
  impl_->base = static_cast<std::byte*>(::operator new(bytes, std::align_val_t{kBaseAlign}));
  impl_->capacity = bytes;
  poison(impl_->base, bytes);
}

Arena::~Arena() = default;

void* Arena::allocate(std::size_t bytes, std::size_t align) {
  if (align == 0 || (align & (align - 1)) != 0) {
    throw std::invalid_argument("Arena::allocate: alignment must be a power of two, got " +
                                std::to_string(align));
  }
  Impl& a = *impl_;
  const auto addr = reinterpret_cast<std::uintptr_t>(a.base) + a.used;
  const auto pad = static_cast<std::size_t>((align - (addr & (align - 1))) & (align - 1));
  // Written as two subtractions so that no request, however large, can wrap.
  const std::size_t free = a.capacity - a.used;
  if (pad > free || bytes > free - pad) {
    a.exhausted(std::to_string(bytes) + " bytes (align " + std::to_string(align) + ")");
  }

  std::byte* p = a.base + a.used + pad;
  a.used += pad + bytes;
  a.peak = std::max(a.peak, a.used);
  unpoison(p, bytes);
  return p;
}

Tensor Arena::tensor(const Shape& shape, DType dtype) {
  // Validate everything Tensor::wrap would before allocating, so a rejected
  // request leaves the arena exactly as it was.
  const std::size_t esize = dtype_size(dtype);  // rejects Q8_0
  if (shape.ndim() < 1) {
    throw std::invalid_argument("Arena::tensor: rank must be 1.." + std::to_string(kMaxDims) +
                                ", got 0");
  }
  for (int i = 0; i < shape.ndim(); ++i) {
    if (shape[i] < 0) {
      throw std::invalid_argument("Arena::tensor: negative extent in shape " + shape.to_string());
    }
  }
  // Checked before multiplying, so numel * esize cannot wrap.
  const auto numel = static_cast<std::size_t>(shape.numel());
  if (numel > impl_->capacity / esize) {
    impl_->exhausted(std::string("tensor ") + dtype_name(dtype) + shape.to_string());
  }
  return Tensor::wrap(allocate(numel * esize), shape, dtype);
}

void Arena::reset() {
  impl_->used = 0;
  poison(impl_->base, impl_->capacity);
}

std::size_t Arena::used() const { return impl_->used; }

std::size_t Arena::capacity() const { return impl_->capacity; }

std::size_t Arena::peak() const { return impl_->peak; }

}  // namespace minillm
