// minillm — runtime.h
//
// Execution plumbing shared by the optimized kernels: a fixed-size thread pool
// and a bump-allocated scratch arena. No work-stealing, no futures, no task
// graph — parallel_for is the only parallel primitive the engine needs.
//
// THIS HEADER IS FROZEN. See docs/ARCHITECTURE.md.

#ifndef MINILLM_RUNTIME_H
#define MINILLM_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

#include "minillm/tensor.h"

namespace minillm {

/// Fixed-size pool of worker threads.
///
/// parallel_for is blocking and must not be called from inside itself (no
/// nested parallelism). A ThreadPool is safe to call parallel_for on from one
/// thread at a time.
class ThreadPool {
 public:
  /// `n_threads` <= 0 means std::thread::hardware_concurrency() (at least 1).
  /// `n_threads == 1` runs everything inline and starts no threads.
  explicit ThreadPool(int n_threads = 0);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  /// Number of threads that execute work, including the calling thread.
  int size() const;

  /// Splits [0, n) into contiguous chunks and calls body(begin, end) once per
  /// chunk, on an unspecified thread, exactly once per chunk, covering the
  /// whole range. Returns when every chunk has finished. An exception thrown
  /// by `body` is rethrown on the calling thread after all chunks complete;
  /// if several throw, one is rethrown and the others are dropped.
  ///
  /// `grain` is the minimum chunk size: fewer than `grain` items run inline
  /// on the calling thread. Pass 1 to let the pool decide the split.
  void parallel_for(std::int64_t n, std::int64_t grain,
                    const std::function<void(std::int64_t begin, std::int64_t end)>& body);

  /// parallel_for with grain = 1.
  void parallel_for(std::int64_t n,
                    const std::function<void(std::int64_t begin, std::int64_t end)>& body);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Bump allocator for per-forward-pass scratch buffers.
///
/// Allocation is a pointer increment; nothing is freed individually. reset()
/// rewinds to empty and is the only way to reclaim. Memory handed out stays
/// valid until reset() or destruction. Not thread-safe.
class Arena {
 public:
  /// Reserves `bytes` up front; the arena never grows.
  explicit Arena(std::size_t bytes);
  ~Arena();

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  /// Returns `bytes` of scratch aligned to `align` (a power of two).
  /// Throws std::bad_alloc if the arena is exhausted — callers are expected
  /// to size the arena for the worst case, not to handle failure.
  void* allocate(std::size_t bytes, std::size_t align = 64);

  /// Typed convenience wrapper. The memory is NOT initialized and no
  /// destructor is ever run, so T must be trivially destructible. Throws
  /// std::bad_alloc if `count * sizeof(T)` does not fit in a size_t, rather
  /// than letting the byte count wrap to a small allocation.
  template <class T>
  T* allocate_array(std::size_t count) {
    static_assert(std::is_trivially_destructible_v<T>, "Arena never runs destructors");
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_alloc();
    return static_cast<T*>(allocate(count * sizeof(T), alignof(T) < 64 ? 64 : alignof(T)));
  }

  /// Non-owning Tensor view over freshly allocated, uninitialized scratch.
  /// The view is valid until the next reset().
  Tensor tensor(const Shape& shape, DType dtype = DType::F32);

  /// Rewinds to empty, invalidating everything handed out so far.
  void reset();

  std::size_t used() const;
  std::size_t capacity() const;
  /// High-water mark of used() since construction, for sizing the arena.
  std::size_t peak() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Number of threads the optimized kernels in minillm::ops may use, counting
/// the calling thread. n <= 0 means std::thread::hardware_concurrency() (at
/// least 1); 1 runs every kernel inline. Blocks until no kernel is running,
/// then resizes the process-wide pool. Default: 0.
void set_num_threads(int n);

/// The resolved setting: always >= 1.
int num_threads();

/// ThreadPool::parallel_for on the process-wide pool, for kernels. Safe to
/// call from any thread, concurrently, and from inside a body: if the pool is
/// busy, the whole range runs inline on the calling thread instead.
void kernel_parallel_for(std::int64_t n, std::int64_t grain,
                         const std::function<void(std::int64_t begin, std::int64_t end)>& body);

}  // namespace minillm

#endif  // MINILLM_RUNTIME_H
