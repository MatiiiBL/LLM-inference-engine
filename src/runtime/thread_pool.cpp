// ThreadPool — owned by workflow 08-runtime.
//
// parallel_for
//
//   1. writes the job (body, n, chunk count) into plain fields,
//   2. publishes it by storing its ticket and chunk count into `unclaimed`,
//   3. hands a ticket to as many workers as the job can use, or may use —
//      the others are not woken at all — and wakes the first of them that
//      is asleep,
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
// Waiting spins for kSpin before it sleeps. A forward pass makes hundreds of
// parallel_for calls back to back, and waking a sleeping thread costs more
// than the work in most of them. Once parallel_for has returned and no new
// call starts, every worker is asleep within kSpin.
//
// Spinning never yields to the OS. On macOS sched_yield() also depresses the
// thread's priority, so on a busy machine a worker that yielded and then
// claimed a chunk could lose its core for a whole scheduler quantum (~10 ms)
// while every other thread waited for that chunk; at the default thread count
// that made decode slower than on one thread. The spin is bounded instead:
// with more runnable threads than cores, a spinner holds a core for at most
// kSpin before it sleeps and frees it.
//
// Waking a sleeper is a system call, and a decode step makes about 240
// parallel_for calls, so the caller does at most one: it wakes the first
// sleeping worker it ticketed, which wakes the next sleeping one before
// running chunks, and so on, for as long as the job has chunks left to claim.
//
// Memory ordering. The job fields are plain. The caller writes them only
// after every chunk of the previous job has finished (it has seen `done`
// reach the chunk count), then release-stores `unclaimed`. A worker reads
// them only after it has claimed a chunk — an acquire read-modify-write on
// `unclaimed` that took one from a positive count — and while it holds that
// unfinished chunk the job cannot complete, so the fields cannot change
// under it. Whichever job a claim lands in, the fields it then reads are
// that job's: a worker woken late for one job may well help with the next.
// Except a strict worker: `unclaimed` carries the job's ticket next to the
// count, and a strict worker claims by compare-exchange, only from the job
// whose ticket it holds. The kernel pool makes strict the workers a small
// call does not use, so a small call never runs on more threads than it
// chose. A claim that comes back empty touches nothing else. Everything a chunk
// does — its body, storing the first exception, destroying any other — comes
// before its fetch_add on `done`. Those increments are all read-modify-writes,
// so the caller's load that sees `done` reach the chunk count synchronizes
// with every one of them, and sees all of it. The two sleep handshakes —
// ticket / sleeping, and done / caller_sleeping — are seq_cst, so of a
// sleeper and its waker at least one sees the other's store: no wakeup is
// lost. A worker that wakes the next sleeper stands in for the caller: the
// caller stored every ticket of the job before it woke the first worker of
// the chain, under that worker's mutex, and each link wakes the next under
// the next one's, so each ticket store happens before every waker's load of
// `sleeping`, as the handshake needs. A worker woken some other way — a
// spurious wakeup, or a late link of an earlier job's chain — may not see
// the later tickets yet and end the chain early; the sleepers after it then
// miss this one job. That costs speed, never progress: the caller claims
// every chunk nobody else does, and only ever waits for chunks already
// claimed, by threads that are running them.

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

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

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

/// `unclaimed` holds a job's ticket in its high 32 bits, so that a strict
/// claim can check the job and the count at once, and kBias plus the job's
/// unclaimed chunks in its low 32. A failed fetch_sub takes the count below
/// zero, but at most once per thread per job, so it never borrows from the
/// ticket. Tickets wrap at 2^32 jobs, which only a strict worker stalled for
/// 2^32 jobs between reading its ticket and claiming could notice.
constexpr std::uint64_t kBias = std::uint64_t{1} << 31;
constexpr std::int64_t kMaxChunks = (std::int64_t{1} << 31) - 1;

constexpr std::uint64_t pack(std::uint64_t ticket, std::int64_t count) {
  return ticket << 32 | (kBias + static_cast<std::uint64_t>(count));
}

/// The unclaimed chunks in `word`, a value of `unclaimed`; < 0 once every
/// chunk is claimed and some claimants have failed.
constexpr std::int64_t unclaimed_count(std::uint64_t word) {
  return static_cast<std::int64_t>(word & 0xffffffffu) - static_cast<std::int64_t>(kBias);
}

/// Whether `word`, a value of `unclaimed`, belongs to the job with `ticket`.
constexpr bool same_job(std::uint64_t word, std::uint64_t ticket) {
  return word >> 32 == (ticket & 0xffffffffu);
}

// True while this thread is running a parallel_for body, on any pool. A
// nested parallel_for would either deadlock (a worker waiting on itself) or
// publish a second job over the one in flight, so it is rejected outright.
// A nested kernel_parallel_for runs inline instead.
thread_local bool t_in_body = false;

// Restores the previous value rather than clearing it: kernel_parallel_for
// runs nested bodies inline, inside the outer body's scope.
class BodyScope {
 public:
  BodyScope() : outer_(t_in_body) { t_in_body = true; }
  ~BodyScope() { t_in_body = outer_; }
  BodyScope(const BodyScope&) = delete;
  BodyScope& operator=(const BodyScope&) = delete;

 private:
  bool outer_;
};

/// One per hardware thread, at least 1: the kernel pool's size for n <= 0.
int hardware_threads() {
  return static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
}

/// One thread per hardware thread of the performance cores, where the OS
/// tells core types apart (Apple silicon's hw.perflevel0, the fastest level),
/// otherwise hardware_threads(). ThreadPool's size for n <= 0, and the most
/// threads a small kernel call uses; see "How many threads a call runs on".
int performance_threads() {
  const int all = hardware_threads();
#if defined(__APPLE__)
  int fast = 0;
  std::size_t size = sizeof fast;
  if (sysctlbyname("hw.perflevel0.logicalcpu", &fast, &size, nullptr, 0) == 0 && fast > 0) {
    return std::min(fast, all);
  }
#endif
  return all;
}

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

/// The workers and the job they share. ThreadPool runs every call on all of
/// them; the kernel pool tells each call how many to use.
struct Pool {
  int n_threads = 1;
  int first_strict = 0;  // workers from this index on are strict; see the top
  std::unique_ptr<Slot[]> slots;  // one per worker, n_threads - 1
  std::vector<std::thread> workers;

  // The job in flight. Plain fields; see the ordering note at the top.
  const Body* body = nullptr;
  std::int64_t n = 0;
  std::int64_t chunks = 0;
  std::exception_ptr error;  // the first exception; written by whoever set `failed`
  std::uint64_t job = 0;     // the last ticket handed out; caller only

  // The current job's ticket and how many of its chunks nobody has claimed
  // yet, as pack() lays them out.
  std::atomic<std::uint64_t> unclaimed{0};
  std::atomic<std::int64_t> done{0};  // chunks of the current job finished
  std::atomic<bool> failed{false};
  std::atomic<bool> busy{false};  // a dispatched job is in flight

  std::mutex done_mu;
  std::condition_variable done_cv;
  std::atomic<bool> caller_sleeping{false};

  // Stops and joins the workers. Lives here rather than in start() so a start
  // that fails halfway through starting threads still joins the ones it did
  // start. Only started workers are sent a stop ticket, so a start that
  // failed to allocate the slots has no slot to touch.
  ~Pool() {
    for (std::size_t w = 0; w < workers.size(); ++w) give_ticket(slots[w], kStop);
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

  /// Waits for a ticket other than `seen` and returns it; `slept` says
  /// whether it took a wakeup.
  static std::uint64_t wait_for_ticket(Slot& s, std::uint64_t seen, bool& slept) {
    std::uint64_t t = seen;
    const auto arrived = [&] {
      t = s.ticket.load();
      return t != seen;
    };
    slept = false;
    if (spin_until(arrived)) return t;
    slept = true;
    std::unique_lock<std::mutex> lock(s.mu);
    s.sleeping.store(true);
    s.cv.wait(lock, arrived);
    s.sleeping.store(false);
    return t;
  }

  /// Wakes the first worker after `index` that holds `ticket` and is asleep,
  /// if any; index -1 starts at the first worker. Workers that are awake are
  /// passed over, since they will see their ticket by themselves. The scan
  /// stops at the first worker without `ticket`: parallel_for hands a job's
  /// tickets to workers 0, 1, 2, ... in order, and a later job's tickets are
  /// that job's to deliver.
  void wake_next(int index, std::uint64_t ticket) {
    for (int w = index + 1; w + 1 < n_threads; ++w) {
      Slot& s = slots[w];
      if (s.ticket.load() != ticket) return;
      if (s.sleeping.load()) {
        { std::lock_guard<std::mutex> lock(s.mu); }
        s.cv.notify_one();
        return;
      }
    }
  }

  /// Claims a chunk of the current job, or for a strict claimant only of the
  /// job with `ticket`. Returns how many chunks were unclaimed before, or 0.
  std::int64_t claim(std::uint64_t ticket, bool strict) noexcept {
    if (!strict) {
      const std::uint64_t word = unclaimed.fetch_sub(1, std::memory_order_acquire);
      return std::max<std::int64_t>(unclaimed_count(word), 0);
    }
    std::uint64_t word = unclaimed.load(std::memory_order_relaxed);
    while (same_job(word, ticket) && unclaimed_count(word) > 0) {
      if (unclaimed.compare_exchange_weak(word, word - 1, std::memory_order_acquire,
                                          std::memory_order_relaxed)) {
        return unclaimed_count(word);
      }
    }
    return 0;
  }

  /// Claims and runs chunks until none are left; see claim().
  void run_chunks(std::uint64_t ticket, bool strict) noexcept {
    for (;;) {
      const std::int64_t left = claim(ticket, strict);
      if (left == 0) return;
      // This chunk is ours and unfinished, so the job and its fields stay put
      // until the done.fetch_add below. Read everything needed before it.
      const std::int64_t total = chunks;
      const std::int64_t c = total - left;
      {
        std::exception_ptr e =
            run_chunk(*body, chunk_begin(n, total, c), chunk_begin(n, total, c + 1));
        if (e && !failed.exchange(true, std::memory_order_relaxed)) error = std::move(e);
        // An exception that lost to another chunk's is destroyed here. Its
        // destructor may touch anything the caller owns, so it has to be
        // over before the chunk counts as done and parallel_for can return.
      }
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
      bool slept = false;
      seen = wait_for_ticket(s, seen, slept);
      if (seen == kStop) return;
      // Pass the wakeup on, unless the job is already all claimed: then the
      // next sleeper would only find nothing to do.
      if (slept) {
        const std::uint64_t word = unclaimed.load(std::memory_order_relaxed);
        if (same_job(word, seen) && unclaimed_count(word) > 0) wake_next(index, seen);
      }
      run_chunks(seen, index >= first_strict);
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

  /// Starts the n_threads - 1 workers; the caller is the n_threads-th. Workers
  /// `strict_from` and later only claim chunks of a job they were ticketed for.
  void start(int threads, int strict_from) {
    n_threads = threads;
    first_strict = strict_from;
    slots = std::make_unique<Slot[]>(static_cast<std::size_t>(threads - 1));
    workers.reserve(static_cast<std::size_t>(threads - 1));
    for (int w = 0; w + 1 < threads; ++w) workers.emplace_back([this, w] { worker_loop(w); });
  }

  /// parallel_for on the calling thread and workers 0 .. threads - 2, with up
  /// to `per_thread` chunks for each. n > 0, 1 <= threads <= n_threads, and
  /// the caller is not inside a body.
  void run(std::int64_t n_items, std::int64_t grain, const Body& job_body, int threads,
           std::int64_t per_thread) {
    // Every chunk gets at least `grain` items, so there are at most n / grain
    // of them. One chunk, or one thread, means no parallelism: run it all here
    // as a single chunk, touching nothing shared.
    grain = std::max<std::int64_t>(grain, 1);
    const std::int64_t total = std::min({n_items / grain, per_thread * threads, kMaxChunks});
    if (total <= 1 || threads == 1) {
      BodyScope scope;
      job_body(0, n_items);
      return;
    }

    if (busy.exchange(true, std::memory_order_acquire)) {
      throw std::logic_error(
          "ThreadPool::parallel_for: called from two threads at once; a pool runs one "
          "parallel_for at a time");
    }
    const std::uint64_t ticket = ++job;
    body = &job_body;
    n = n_items;
    chunks = total;
    done.store(0, std::memory_order_relaxed);
    failed.store(false, std::memory_order_relaxed);
    unclaimed.store(pack(ticket, total), std::memory_order_release);

    // Every ticket is stored before anyone is woken; see wake_next. Workers
    // past the job's last helper keep their old ticket and are not woken.
    const int helpers = static_cast<int>(std::min<std::int64_t>(total, threads) - 1);
    for (int w = 0; w < helpers; ++w) slots[w].ticket.store(ticket);
    wake_next(-1, ticket);

    run_chunks(ticket, false);
    // Every chunk, even after one has thrown: a running chunk still uses `body`.
    wait_for_chunks(total);

    std::exception_ptr e = std::move(error);
    error = nullptr;
    body = nullptr;
    busy.store(false, std::memory_order_release);
    if (e) std::rethrow_exception(e);
  }
};

}  // namespace

struct ThreadPool::Impl : Pool {};

ThreadPool::ThreadPool(int n_threads) : impl_(std::make_unique<Impl>()) {
  const int threads = n_threads > 0 ? n_threads : performance_threads();
  impl_->start(threads, threads);
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
  impl_->run(n, grain, body, impl_->n_threads, kChunksPerThread);
}

void ThreadPool::parallel_for(std::int64_t n, const Body& body) { parallel_for(n, 1, body); }

// ---------------------------------------------------------------------------
// The process-wide pool behind kernel_parallel_for
//
// ThreadPool::parallel_for may be entered by one thread at a time and never
// from inside a body, while ops.h promises that ops are safe to call
// concurrently. A mutex bridges the two: a kernel try-locks it and, if it is
// taken — another kernel holds the pool, or set_num_threads is swapping it —
// runs its whole range inline. Kernels never wait for one another, so they
// cannot deadlock, and set_num_threads, which does lock it, waits only for
// the one kernel that holds it.
//
// A call from inside a body runs inline without touching the mutex at all:
// on the thread that holds it, try_lock would be undefined behaviour.
//
// How many threads a call runs on. The pool has num_threads() threads, by
// default one per hardware thread. Where the performance cores are not all
// of them (Apple silicon), a call uses at most the performance cores'
// hardware threads unless it says, with a work size of at least kLargeWork,
// that it is large. The OS still decides which core runs which thread; four
// busy threads on an M4 land on its four performance cores.
//
// Large calls are compute-bound: a 128-token prefill ran 1.6-1.9x faster on
// all 10 cores of an M4 than on its 4 performance cores. Decode's calls are
// small and stream their weights, which 4 performance cores already do at
// the bandwidth of 10; on a busy machine the extra, slower, more often
// preempted threads made decode slower. Putting decode's largest calls
// (gate/up/down_proj, 0.88M multiply-adds) on every core too made it 7%
// slower as found and 4% slower with every core busy. The lm_head, at 28M,
// is the exception: decode with it on every core was within 2% as found of
// decode with it on the performance cores, and 5% faster with every core
// busy.
//
// kLargeWork was chosen on the M4 by sweeping it over prompts of 4 to 32
// tokens, where the projections range from 0.4M to 28M multiply-adds:
// whole-prefill speed against the 4-thread pool, interleaved in one batch,
//
//     threshold     4 tokens   8 tokens   16 tokens   32 tokens
//     1M            1.23x      1.15x      1.27x       1.35x
//     2M            1.19x      1.16x      1.23x       1.35x
//     4M            1.05x      1.13x      1.23x       1.33x
//     8M            1.05x      1.04x      1.16x       1.34x
//
// so 1M (2^20). Every decode call of SmolLM2-135M but the lm_head stays
// below it: the largest, gate/up/down_proj, are 0.88M.
//
// Chunks per thread. A large call gets kLargeChunksPerThread: its workers
// start at different times, the efficiency cores' up to tens of
// microseconds late as the wakeups pass down the chain, and run at different
// speeds, so smaller chunks even out the end of the call; 8 instead of 4
// made a 128-token prefill 8% faster (16 did no better). A small call gets
// kSmallChunksPerThread: a decode step's calls last 2-16 us, and two fewer
// claims per thread per call made decode 3-6% faster on Q8_0 weights and
// 0-5% on F32. A call without a work size keeps the ThreadPool split.
// ---------------------------------------------------------------------------

namespace {

/// Work, in multiply-adds, from which a call may use every thread; see above.
constexpr std::int64_t kLargeWork = std::int64_t{1} << 20;

/// Chunks per thread for a call of at least kLargeWork, and for one that
/// gives a smaller work size; see above.
constexpr std::int64_t kLargeChunksPerThread = 8;
constexpr std::int64_t kSmallChunksPerThread = 2;

int resolve_threads(int n) { return n > 0 ? n : hardware_threads(); }

/// A started kernel pool of `threads` threads. The workers that a call below
/// kLargeWork does not use are strict, so it never runs on more than
/// `small_threads`. If starting one fails, the ones already started are joined.
std::unique_ptr<Pool> make_pool(int threads, int small_threads) {
  auto pool = std::make_unique<Pool>();
  pool->start(threads, std::min(threads, small_threads) - 1);
  return pool;
}

struct KernelPool {
  std::mutex mu;  // held by the kernel using `pool`, or by set_num_threads
  std::unique_ptr<Pool> pool;  // under `mu`; null until first needed, or at 1 thread
  std::atomic<int> threads{resolve_threads(0)};  // written under `mu`
  // The most threads a call below kLargeWork uses: the performance cores'
  // where they are not all of them, otherwise no limit beyond `threads`.
  const int small_threads = performance_threads() < hardware_threads()
                                ? performance_threads()
                                : std::numeric_limits<int>::max();
};

// Never destroyed, so a kernel called from another static's destructor still
// finds it. Its workers are blocked on a condition variable at exit.
KernelPool& kernel_pool() {
  static KernelPool* const kp = new KernelPool;
  return *kp;
}

void run_inline(std::int64_t n, const Body& body) {
  BodyScope scope;
  body(0, n);
}

}  // namespace

void set_num_threads(int n) {
  if (t_in_body) {
    // The kernel this body belongs to holds the mutex and is waiting on the
    // body: locking it here could never succeed.
    throw std::logic_error("set_num_threads: called from inside a parallel_for body");
  }
  const int threads = resolve_threads(n);
  KernelPool& kp = kernel_pool();
  const std::lock_guard<std::mutex> lock(kp.mu);
  if (threads == kp.threads.load(std::memory_order_relaxed)) return;
  kp.pool.reset();
  if (threads > 1) kp.pool = make_pool(threads, kp.small_threads);
  kp.threads.store(threads, std::memory_order_relaxed);
}

int num_threads() { return kernel_pool().threads.load(std::memory_order_relaxed); }

void kernel_parallel_for(std::int64_t n, std::int64_t grain, const Body& body) {
  kernel_parallel_for(n, grain, 0, body);
}

void kernel_parallel_for(std::int64_t n, std::int64_t grain, std::int64_t work, const Body& body) {
  if (n < 0) {
    throw std::invalid_argument("kernel_parallel_for: n must be >= 0, got " + std::to_string(n));
  }
  if (n == 0) return;
  KernelPool& kp = kernel_pool();
  // What Pool::run would run inline anyway, decided here so that small calls
  // never touch the mutex.
  if (t_in_body || n / std::max<std::int64_t>(grain, 1) <= 1 ||
      kp.threads.load(std::memory_order_relaxed) == 1) {
    run_inline(n, body);
    return;
  }
  std::unique_lock<std::mutex> lock(kp.mu, std::try_to_lock);
  if (!lock.owns_lock()) {
    run_inline(n, body);
    return;
  }
  const int threads = kp.threads.load(std::memory_order_relaxed);
  if (threads == 1) {
    run_inline(n, body);
    return;
  }
  if (kp.pool == nullptr) kp.pool = make_pool(threads, kp.small_threads);
  if (work >= kLargeWork) {
    kp.pool->run(n, grain, body, threads, kLargeChunksPerThread);
  } else {
    kp.pool->run(n, grain, body, std::min(threads, kp.small_threads),
                 work > 0 ? kSmallChunksPerThread : kChunksPerThread);
  }
}

}  // namespace minillm
