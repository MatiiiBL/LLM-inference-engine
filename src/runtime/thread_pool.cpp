// ThreadPool — owned by workflow 08-runtime.
//
// STUB. Construction succeeds and size() reports 1 so that code holding a
// pool can be written and tested before the pool exists; parallel_for throws.

#include <stdexcept>

#include "minillm/runtime.h"

namespace minillm {

struct ThreadPool::Impl {
  int n_threads = 1;
};

ThreadPool::ThreadPool(int) : impl_(std::make_unique<Impl>()) {}

ThreadPool::~ThreadPool() = default;

int ThreadPool::size() const { return impl_->n_threads; }

void ThreadPool::parallel_for(std::int64_t, std::int64_t,
                              const std::function<void(std::int64_t, std::int64_t)>&) {
  throw std::logic_error("not implemented: ThreadPool::parallel_for (workflow 08-runtime)");
}

void ThreadPool::parallel_for(std::int64_t n,
                              const std::function<void(std::int64_t, std::int64_t)>& body) {
  parallel_for(n, 1, body);
}

}  // namespace minillm
