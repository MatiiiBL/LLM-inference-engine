// ThreadPool — owned by workflow 08-runtime.
//
// One mutex, two condition variables, one job in flight at a time.
//
// parallel_for publishes the job under the lock and bumps a generation
// counter. [0, n) is split into `chunks` contiguous pieces; the caller runs
// chunk 0 itself and worker w runs chunk w + 1, so a job never has more chunks
// than size(). Each worker that had a chunk counts itself out of `pending`,
// and the caller waits for that count to reach zero before it returns.
//
// A worker cannot miss a job it has a chunk in: the caller does not publish
// the next job until every worker of the current one has counted out, and a
// worker with no chunk in a job has nothing to miss.

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "minillm/runtime.h"

namespace minillm {
namespace {

using Body = std::function<void(std::int64_t, std::int64_t)>;

// True while this thread is running a parallel_for body, on any pool. A
// nested parallel_for would either deadlock (a worker waiting on itself) or
// publish a second job over the one in flight, so it is rejected outright.
thread_local bool t_in_body = false;

class BodyScope {
 public:
  BodyScope() { t_in_body = true; }
  ~BodyScope() { t_in_body = false; }
  BodyScope(const BodyScope&) = delete;
  BodyScope& operator=(const BodyScope&) = delete;
};

/// Start of chunk `i` when [0, n) is cut into `chunks` contiguous pieces whose
/// sizes differ by at most one. chunk_begin(n, chunks, chunks) == n.
std::int64_t chunk_begin(std::int64_t n, std::int64_t chunks, std::int64_t i) {
  return i * (n / chunks) + std::min(i, n % chunks);
}

/// Runs one chunk and captures anything it throws, so no exception ever
/// unwinds through the pool's own bookkeeping.
std::exception_ptr run_chunk(const Body& body, std::int64_t begin, std::int64_t end) noexcept {
  BodyScope scope;
  try {
    body(begin, end);
  } catch (...) {
    return std::current_exception();
  }
  return nullptr;
}

}  // namespace

struct ThreadPool::Impl {
  int n_threads = 1;
  std::vector<std::thread> workers;

  std::mutex mu;
  std::condition_variable work_cv;  // workers: a job was published, or stop
  std::condition_variable done_cv;  // caller: the last worker chunk finished

  // Everything below is guarded by mu.
  std::uint64_t generation = 0;
  bool stop = false;
  bool busy = false;  // a dispatched job is in flight
  const Body* body = nullptr;
  std::int64_t n = 0;
  std::int64_t chunks = 0;
  int pending = 0;  // worker chunks of the current job not yet finished
  std::exception_ptr error;

  // Stops and joins the workers. Lives here rather than in ~ThreadPool so a
  // constructor that fails halfway through starting threads still joins the
  // ones it did start.
  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mu);
      stop = true;
    }
    work_cv.notify_all();
    for (std::thread& t : workers) t.join();
  }

  void worker_loop(int index) {
    const std::int64_t chunk = index + 1;
    std::uint64_t seen = 0;
    std::unique_lock<std::mutex> lock(mu);
    for (;;) {
      work_cv.wait(lock, [&] { return stop || generation != seen; });
      if (stop) return;
      seen = generation;
      if (chunk >= chunks) continue;

      const Body& f = *body;
      const std::int64_t begin = chunk_begin(n, chunks, chunk);
      const std::int64_t end = chunk_begin(n, chunks, chunk + 1);
      lock.unlock();
      std::exception_ptr e = run_chunk(f, begin, end);
      lock.lock();

      if (e && !error) error = std::move(e);
      if (--pending == 0) done_cv.notify_one();
    }
  }
};

ThreadPool::ThreadPool(int n_threads) : impl_(std::make_unique<Impl>()) {
  if (n_threads <= 0) {
    n_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  }
  impl_->n_threads = n_threads;
  impl_->workers.reserve(static_cast<std::size_t>(n_threads - 1));
  for (int w = 0; w + 1 < n_threads; ++w) {
    impl_->workers.emplace_back([p = impl_.get(), w] { p->worker_loop(w); });
  }
}

ThreadPool::~ThreadPool() = default;

int ThreadPool::size() const { return impl_->n_threads; }

void ThreadPool::parallel_for(std::int64_t n, std::int64_t grain, const Body& body) {
  if (n < 0) {
    throw std::invalid_argument("ThreadPool::parallel_for: n must be >= 0, got " +
                                std::to_string(n));
  }
  if (t_in_body) {
    throw std::logic_error(
        "ThreadPool::parallel_for: called from inside a parallel_for body; nested parallelism "
        "is not supported");
  }
  if (n == 0) return;

  // Every chunk gets at least `grain` items, so there are at most n / grain
  // of them. One chunk means no parallelism: run it here, touching nothing
  // shared.
  grain = std::max<std::int64_t>(grain, 1);
  const std::int64_t chunks = std::min<std::int64_t>(impl_->n_threads, n / grain);
  if (chunks <= 1) {
    BodyScope scope;
    body(0, n);
    return;
  }

  Impl& p = *impl_;
  {
    std::lock_guard<std::mutex> lock(p.mu);
    if (p.busy) {
      throw std::logic_error(
          "ThreadPool::parallel_for: called from two threads at once; a pool runs one "
          "parallel_for at a time");
    }
    p.busy = true;
    p.body = &body;
    p.n = n;
    p.chunks = chunks;
    p.pending = static_cast<int>(chunks - 1);
    ++p.generation;
  }
  p.work_cv.notify_all();

  std::exception_ptr error = run_chunk(body, 0, chunk_begin(n, chunks, 1));

  // Wait even if chunk 0 threw: the workers still hold a reference to `body`.
  {
    std::unique_lock<std::mutex> lock(p.mu);
    p.done_cv.wait(lock, [&] { return p.pending == 0; });
    if (!error) error = std::move(p.error);
    p.error = nullptr;
    p.body = nullptr;
    p.busy = false;
  }
  if (error) std::rethrow_exception(error);
}

void ThreadPool::parallel_for(std::int64_t n, const Body& body) { parallel_for(n, 1, body); }

}  // namespace minillm
