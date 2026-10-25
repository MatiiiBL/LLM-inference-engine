// ThreadPool — owned by workflow 08-runtime.
//
// parallel_for
//
//   1. writes the job (body, n, chunk count) into plain fields,
//   2. publishes it by storing the chunk count into `unclaimed`,
//   3. hands a ticket to as many workers as the job can use — the others are
//      not woken at all,
//   4. claims chunks alongside them, and
//   5. waits until every chunk has finished.
//
// It waits for chunks, not for workers. A ticketed worker that the OS has not
// scheduled yet (there are more runnable threads than cores) simply finds
// nothing left to claim when it does run; the call does not stall on it.
//
// Chunks are claimed, not assigned, so a fast core ends up running more of
// them than a slow one. On a machine with performance and efficiency cores a
// static one-chunk-per-thread split runs at the speed of the slowest core.
//
// Waiting spins for a short while, yielding the core to anything else that
// is runnable, before it sleeps. A forward pass makes hundreds of
// parallel_for calls back to back, and waking a sleeping thread costs more
// than the work in most of them.
//
// Memory ordering. The job fields are plain. The caller writes them only
// after every chunk of the previous job has finished (it has seen `done`
// reach the chunk count), then release-stores `unclaimed`. A worker reads
// them only after it has claimed a chunk — an acquire fetch_sub on
// `unclaimed` that returned a positive count — and while it holds that
// unfinished chunk the job cannot complete, so the fields cannot change
// under it. Whichever job a claim lands in, the fields it then reads are
// that job's: a worker woken late for one job may well help with the next.
// A claim that comes back empty touches nothing else. The two sleep
// handshakes — ticket / sleeping, and done / caller_sleeping — are seq_cst,
// so of a sleeper and its waker at least one sees the other's store: no
// wakeup is lost.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "minillm/runtime.h"

namespace minillm {
namespace {

using Body = std::function<void(std::int64_t, std::int64_t)>;

/// How long a waiting thread spins before it goes to sleep.
constexpr std::chrono::microseconds kSpin{50};

/// Chunks per thread when n allows it: enough to even out uneven cores,
/// few enough that claiming them costs nothing measurable.
constexpr std::int64_t kChunksPerThread = 4;

/// The ticket that tells a worker to exit.
constexpr std::uint64_t kStop = std::numeric_limits<std::uint64_t>::max();

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

inline void cpu_relax() {
#if defined(__aarch64__)
  __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#endif
}

/// Spins until ready() or kSpin has passed, and returns the last ready().
/// Yields between bursts: with more runnable threads than cores, a spinner
/// must not keep the core from a thread that is holding a chunk.
template <class Ready>
bool spin_until(Ready&& ready) {
  using clock = std::chrono::steady_clock;
  const auto deadline = clock::now() + kSpin;
  for (;;) {
    for (int i = 0; i < 64; ++i) {
      if (ready()) return true;
      cpu_relax();
    }
    if (clock::now() >= deadline) return ready();
    std::this_thread::yield();
  }
}

/// One worker's mailbox. Cache-line aligned so that handing out tickets
/// does not bounce one line between every core.
struct alignas(64) Slot {
  std::atomic<std::uint64_t> ticket{0};
  std::atomic<bool> sleeping{false};
  std::mutex mu;
  std::condition_variable cv;
};

}  // namespace

struct ThreadPool::Impl {
  int n_threads = 1;
  std::unique_ptr<Slot[]> slots;  // one per worker, n_threads - 1
  std::vector<std::thread> workers;

  // The job in flight. Plain fields; see the ordering note at the top.
  const Body* body = nullptr;
  std::int64_t n = 0;
  std::int64_t chunks = 0;
  std::exception_ptr error;  // the first exception; written by whoever set `failed`
  std::uint64_t job = 0;     // the last ticket handed out; caller only

  // Chunks of the current job nobody has claimed. Every claimant takes one
  // more than exists and so leaves it negative; the next job's store resets it.
  std::atomic<std::int64_t> unclaimed{0};
  std::atomic<std::int64_t> done{0};  // chunks of the current job finished
  std::atomic<bool> failed{false};
  std::atomic<bool> busy{false};  // a dispatched job is in flight

  std::mutex done_mu;
  std::condition_variable done_cv;
  std::atomic<bool> caller_sleeping{false};

  // Stops and joins the workers. Lives here rather than in ~ThreadPool so a
  // constructor that fails halfway through starting threads still joins the
  // ones it did start.
  ~Impl() {
    for (int w = 0; w + 1 < n_threads; ++w) give_ticket(slots[w], kStop);
    for (std::thread& t : workers) t.join();
  }

  static void give_ticket(Slot& s, std::uint64_t ticket) {
    s.ticket.store(ticket);
    if (s.sleeping.load()) {
      // Taking the lock means the worker is either asleep, or has not yet
      // checked its ticket under the lock and will see this one.
      { std::lock_guard<std::mutex> lock(s.mu); }
      s.cv.notify_one();
    }
  }

  static std::uint64_t wait_for_ticket(Slot& s, std::uint64_t seen) {
    std::uint64_t t = seen;
    const auto arrived = [&] {
      t = s.ticket.load();
      return t != seen;
    };
    if (spin_until(arrived)) return t;
    std::unique_lock<std::mutex> lock(s.mu);
    s.sleeping.store(true);
    s.cv.wait(lock, arrived);
    s.sleeping.store(false);
    return t;
  }

  /// Claims and runs chunks until none are left.
  void run_chunks() noexcept {
    for (;;) {
      const std::int64_t left = unclaimed.fetch_sub(1, std::memory_order_acquire);
      if (left <= 0) return;
      // This chunk is ours and unfinished, so the job and its fields stay put
      // until the done.fetch_add below. Read everything needed before it.
      const std::int64_t total = chunks;
      const std::int64_t c = total - left;
      std::exception_ptr e =
          run_chunk(*body, chunk_begin(n, total, c), chunk_begin(n, total, c + 1));
      if (e && !failed.exchange(true, std::memory_order_relaxed)) error = std::move(e);
      if (done.fetch_add(1) + 1 == total && caller_sleeping.load()) {
        { std::lock_guard<std::mutex> lock(done_mu); }
        done_cv.notify_one();
      }
    }
  }

  void worker_loop(int index) {
    Slot& s = slots[index];
    std::uint64_t seen = 0;
    for (;;) {
      seen = wait_for_ticket(s, seen);
      if (seen == kStop) return;
      run_chunks();
    }
  }

  void wait_for_chunks(std::int64_t total) {
    const auto finished = [&] { return done.load() == total; };
    if (spin_until(finished)) return;
    std::unique_lock<std::mutex> lock(done_mu);
    caller_sleeping.store(true);
    done_cv.wait(lock, finished);
    caller_sleeping.store(false);
  }
};

ThreadPool::ThreadPool(int n_threads) : impl_(std::make_unique<Impl>()) {
  if (n_threads <= 0) {
    n_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  }
  impl_->n_threads = n_threads;
  impl_->slots = std::make_unique<Slot[]>(static_cast<std::size_t>(n_threads - 1));
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
  // of them. One chunk, or one thread, means no parallelism: run it all here
  // as a single chunk, touching nothing shared.
  Impl& p = *impl_;
  grain = std::max<std::int64_t>(grain, 1);
  const std::int64_t chunks = std::min(n / grain, kChunksPerThread * p.n_threads);
  if (chunks <= 1 || p.n_threads == 1) {
    BodyScope scope;
    body(0, n);
    return;
  }

  if (p.busy.exchange(true, std::memory_order_acquire)) {
    throw std::logic_error(
        "ThreadPool::parallel_for: called from two threads at once; a pool runs one "
        "parallel_for at a time");
  }
  p.body = &body;
  p.n = n;
  p.chunks = chunks;
  p.done.store(0, std::memory_order_relaxed);
  p.failed.store(false, std::memory_order_relaxed);
  p.unclaimed.store(chunks, std::memory_order_release);

  const std::uint64_t ticket = ++p.job;
  const int helpers = static_cast<int>(std::min<std::int64_t>(chunks, p.n_threads) - 1);
  for (int w = 0; w < helpers; ++w) Impl::give_ticket(p.slots[w], ticket);

  p.run_chunks();
  // Every chunk, even after one has thrown: a running chunk still uses `body`.
  p.wait_for_chunks(chunks);

  std::exception_ptr error = std::move(p.error);
  p.error = nullptr;
  p.body = nullptr;
  p.busy.store(false, std::memory_order_release);
  if (error) std::rethrow_exception(error);
}

void ThreadPool::parallel_for(std::int64_t n, const Body& body) { parallel_for(n, 1, body); }

}  // namespace minillm
