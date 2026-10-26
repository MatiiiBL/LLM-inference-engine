// Workflow 08-runtime owns this file.
//
// ThreadPool and Arena are infrastructure, not numerics: there is no oracle
// data, only the contracts in runtime.h. Run this under the tsan preset as
// well as asan — a race in the pool shows up as a flaky numeric failure in
// some other module, never here, unless TSan is watching.
//
// Unless a test is about synchronization itself, parallel_for bodies write
// plain ints into disjoint slots rather than taking a lock or using atomics.
// Test-side synchronization would give TSan happens-before edges that hide a
// missing one in the pool.

#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <filesystem>
#endif

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define MINILLM_TEST_ASAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define MINILLM_TEST_ASAN 1
#endif
#ifdef MINILLM_TEST_ASAN
#include <sanitizer/asan_interface.h>
#endif

#include "common/oracle.h"
#include "minillm/runtime.h"

using namespace minillm;
using namespace minillm::test;

namespace {

using Index = std::int64_t;

bool aligned(const void* p, std::size_t align) {
  return reinterpret_cast<std::uintptr_t>(p) % align == 0;
}

/// Threads in this process according to the OS, or -1 where this test does
/// not know how to ask.
int os_thread_count() {
#if defined(__APPLE__)
  thread_act_array_t threads = nullptr;
  mach_msg_type_number_t count = 0;
  if (task_threads(mach_task_self(), &threads, &count) != KERN_SUCCESS) return -1;
  for (mach_msg_type_number_t i = 0; i < count; ++i) {
    mach_port_deallocate(mach_task_self(), threads[i]);
  }
  vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(threads),
                sizeof(thread_act_t) * count);
  return static_cast<int>(count);
#elif defined(__linux__)
  int count = 0;
  for (const auto& entry : std::filesystem::directory_iterator("/proc/self/task")) {
    (void)entry;
    ++count;
  }
  return count;
#else
  return -1;
#endif
}

/// os_thread_count() once it has stopped changing. A joined thread can stay
/// visible to the OS for a moment while the kernel finishes tearing it down.
int settled_thread_count() {
  int last = os_thread_count();
  for (int i = 0; i < 1000; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const int now = os_thread_count();
    if (now == last) return now;
    last = now;
  }
  return last;
}

/// What n <= 0 resolves to, worked out independently of the pool: the
/// hardware threads of the performance cores where the OS reports them
/// (Apple silicon's hw.perflevel0), otherwise std::thread::hardware_concurrency().
int expected_default_threads() {
  const int all = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
#if defined(__APPLE__)
  int fast = 0;
  std::size_t size = sizeof fast;
  if (sysctlbyname("hw.perflevel0.logicalcpu", &fast, &size, nullptr, 0) == 0 && fast > 0) {
    return std::min(fast, all);
  }
#endif
  return all;
}

/// CPU time used so far by every thread of this process.
std::chrono::nanoseconds process_cpu_time() {
  timespec ts{};
  REQUIRE(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) == 0);
  return std::chrono::seconds(ts.tv_sec) + std::chrono::nanoseconds(ts.tv_nsec);
}

/// What one parallel_for call did, as observed by check_cover.
struct Cover {
  Index chunks = 0;
  std::set<std::thread::id> threads;
};

using Body = std::function<void(Index, Index)>;
using ParallelFor = std::function<void(Index n, Index grain, const Body& body)>;

/// Runs parallel_for(n, grain) and checks the split it produced: every index
/// in [0, n) covered exactly once, every chunk non-empty, and every chunk at
/// least `grain` long unless it is the whole range. `size` is the most
/// threads it may run on.
Cover check_cover(const ParallelFor& parallel_for, int size, Index n, Index grain) {
  CAPTURE(size);
  CAPTURE(n);
  CAPTURE(grain);
  const auto un = static_cast<std::size_t>(n);
  std::vector<int> hits(un, 0);
  // Keyed by the chunk's first index, so each chunk writes its own slot.
  std::vector<Index> chunk_end(un, -1);
  std::vector<std::thread::id> chunk_thread(un);
  std::atomic<bool> out_of_range{false};  // only ever written on failure

  parallel_for(n, grain, [&](Index begin, Index end) {
    if (begin < 0 || end > n || begin >= end) {
      out_of_range.store(true, std::memory_order_relaxed);
      return;
    }
    for (Index i = begin; i < end; ++i) ++hits[static_cast<std::size_t>(i)];
    chunk_end[static_cast<std::size_t>(begin)] = end;
    chunk_thread[static_cast<std::size_t>(begin)] = std::this_thread::get_id();
  });

  CHECK_FALSE(out_of_range.load());
  const auto bad = std::find_if(hits.begin(), hits.end(), [](int h) { return h != 1; });
  if (bad != hits.end()) {
    FAIL_CHECK("index " << (bad - hits.begin()) << " was covered " << *bad << " times");
  }

  Cover cover;
  Index pos = 0;
  while (pos < n) {
    const Index end = chunk_end[static_cast<std::size_t>(pos)];
    CAPTURE(pos);
    REQUIRE(end > pos);  // a chunk starts wherever the previous one ended
    if (end - pos < grain) CHECK((pos == 0 && end == n));
    cover.threads.insert(chunk_thread[static_cast<std::size_t>(pos)]);
    ++cover.chunks;
    pos = end;
  }
  CHECK(cover.threads.size() <= static_cast<std::size_t>(size));
  return cover;
}

Cover check_cover(ThreadPool& pool, Index n, Index grain) {
  return check_cover(
      [&](Index n_, Index grain_, const Body& body) { pool.parallel_for(n_, grain_, body); },
      pool.size(), n, grain);
}

/// check_cover for kernel_parallel_for at the current num_threads().
Cover check_kernel_cover(Index n, Index grain) {
  return check_cover(kernel_parallel_for, num_threads(), n, grain);
}

/// Sets the kernel thread count for one test and restores it afterwards,
/// pass or fail.
class KernelThreads {
 public:
  explicit KernelThreads(int n) : saved_(num_threads()) { set_num_threads(n); }
  ~KernelThreads() { set_num_threads(saved_); }
  KernelThreads(const KernelThreads&) = delete;
  KernelThreads& operator=(const KernelThreads&) = delete;

 private:
  int saved_;
};

/// The counts the kernel tests sweep. 10 is the development machine's
/// hardware_concurrency(); on a smaller machine it oversubscribes, which
/// tests the split just as well.
constexpr int kThreadCounts[] = {1, 2, 3, 4, 10};

}  // namespace

// ---------------------------------------------------------------------------
// ThreadPool
// ---------------------------------------------------------------------------

TEST_CASE("ThreadPool size counts the calling thread") {
  const int def = expected_default_threads();
  CHECK(ThreadPool(1).size() == 1);
  CHECK(ThreadPool(2).size() == 2);
  CHECK(ThreadPool(5).size() == 5);
  CHECK(ThreadPool(0).size() == def);
  CHECK(ThreadPool(-3).size() == def);
  CHECK(ThreadPool().size() == def);
}

TEST_CASE("the default thread count is the performance cores, or every hardware thread") {
  const int hw = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  const int def = expected_default_threads();
  MESSAGE("default thread count " << def << " of " << hw << " hardware threads");
  CHECK(def >= 1);
  CHECK(def <= hw);
#if !defined(__APPLE__)
  CHECK(def == hw);  // no core types to tell apart: unchanged from hardware_concurrency()
#endif
}

TEST_CASE("parallel_for covers [0, n) exactly once") {
  for (const int threads : {1, 2, 3, 8}) {
    ThreadPool pool(threads);
    for (const Index n : {0, 1, 2, 7, 1000, 1'000'000}) {
      CAPTURE(threads);
      CAPTURE(n);
      if (n == 0) {
        int calls = 0;
        pool.parallel_for(0, [&](Index, Index) { ++calls; });
        pool.parallel_for(0, 1000, [&](Index, Index) { ++calls; });
        CHECK(calls == 0);
        continue;
      }
      check_cover(pool, n, 1);
    }
  }
}

TEST_CASE("parallel_for without a grain covers the range too") {
  ThreadPool pool(4);
  std::vector<int> hits(1001, 0);
  pool.parallel_for(1001, [&](Index begin, Index end) {
    for (Index i = begin; i < end; ++i) ++hits[static_cast<std::size_t>(i)];
  });
  CHECK(std::count(hits.begin(), hits.end(), 1) == 1001);
}

TEST_CASE("grain is the minimum chunk size") {
  struct Case {
    Index n, grain;
  };
  for (const int threads : {1, 2, 3, 8}) {
    ThreadPool pool(threads);
    for (const Case c : {Case{10, 1000}, Case{10, 4}, Case{10, 10}, Case{1000, 300},
                         Case{1000, 999}, Case{1000, 1000}, Case{1'000'000, 4096}}) {
      check_cover(pool, c.n, c.grain);
    }
  }
}

TEST_CASE("a grain below 1 behaves like grain 1") {
  ThreadPool pool(3);
  check_cover(pool, 100, 0);
  check_cover(pool, 100, -7);
}

TEST_CASE("fewer than grain items run inline on the calling thread") {
  for (const int threads : {1, 2, 8}) {
    ThreadPool pool(threads);
    int calls = 0;
    Index b = -1;
    Index e = -1;
    std::thread::id where;
    pool.parallel_for(10, 1000, [&](Index begin, Index end) {
      ++calls;
      b = begin;
      e = end;
      where = std::this_thread::get_id();
    });
    CAPTURE(threads);
    CHECK(calls == 1);
    CHECK(b == 0);
    CHECK(e == 10);
    CHECK(where == std::this_thread::get_id());
  }
}

TEST_CASE("a pool of size 1 starts no threads and runs everything inline") {
  std::thread([] {}).join();  // start any thread a sanitizer runtime starts lazily
  const int before = settled_thread_count();

  ThreadPool pool(1);
  for (const Index n : {1, 7, 1'000'000}) {
    const Cover cover = check_cover(pool, n, 1);
    CHECK(cover.chunks == 1);
    CHECK(cover.threads == std::set<std::thread::id>{std::this_thread::get_id()});
  }

  if (before < 0) {
    note_skip("runtime: no OS thread count on this platform; size-1 pool thread check skipped");
    MESSAGE("SKIP: no OS thread count on this platform");
    return;
  }
  CHECK(settled_thread_count() == before);
}

TEST_CASE("threads are started once, in the constructor, and reused") {
  // Thread ids cannot show this on their own: the OS may give a joined
  // thread's id to the next thread started, so a pool that started and
  // joined a thread in every call would still show no more than size() ids.
  // So every call also asks the OS how many threads the process has, from
  // inside the bodies while all size() items are running at once, and that
  // must be the count right after construction.
  using clock = std::chrono::steady_clock;
  std::thread([] {}).join();  // start any thread a sanitizer runtime starts lazily
  for (const int threads : {2, 3, 8}) {
    CAPTURE(threads);
    const int before = settled_thread_count();
    ThreadPool pool(threads);
    const int started = os_thread_count();

    std::set<std::thread::id> seen;
    std::vector<std::thread::id> slot(static_cast<std::size_t>(threads));
    std::vector<int> during(static_cast<std::size_t>(threads));
    int miscounted = 0;  // samples taken during a call that differ from `started`
    for (int call = 0; call < 200; ++call) {
      std::atomic<int> arrived{0};
      pool.parallel_for(threads, [&](Index begin, Index end) {
        for (Index i = begin; i < end; ++i) {
          slot[static_cast<std::size_t>(i)] = std::this_thread::get_id();
          // Sample only once every item has started, so that every thread
          // the call runs on is alive, and running, at that moment.
          arrived.fetch_add(1);
          const auto deadline = clock::now() + std::chrono::seconds(10);
          while (arrived.load() < threads && clock::now() < deadline) std::this_thread::yield();
          during[static_cast<std::size_t>(i)] = os_thread_count();
        }
      });
      seen.insert(slot.begin(), slot.end());
      miscounted += static_cast<int>(threads - std::count(during.begin(), during.end(), started));
    }
    // Bounded by size(): the same threads every time, as far as ids can tell.
    CHECK(seen.size() <= static_cast<std::size_t>(threads));

    if (before >= 0) {
      CHECK(started == before + threads - 1);
      CHECK(miscounted == 0);
      CHECK(settled_thread_count() == started);
    }
  }
  if (os_thread_count() < 0) {
    note_skip("runtime: no OS thread count on this platform; thread start checks skipped");
    MESSAGE("SKIP: no OS thread count on this platform");
  }
}

TEST_CASE("chunks run concurrently, one per thread") {
  // Every item waits until all of them have started. That can only finish if
  // size() chunks really are in flight at once, one on each thread.
  using clock = std::chrono::steady_clock;
  for (const int threads : {2, 3, 8}) {
    CAPTURE(threads);
    ThreadPool pool(threads);
    std::atomic<int> arrived{0};
    std::vector<char> met(static_cast<std::size_t>(threads), 0);
    pool.parallel_for(threads, [&](Index begin, Index end) {
      for (Index i = begin; i < end; ++i) {
        arrived.fetch_add(1);
        const auto deadline = clock::now() + std::chrono::seconds(10);
        while (arrived.load() < threads && clock::now() < deadline) std::this_thread::yield();
        met[static_cast<std::size_t>(i)] = arrived.load() >= threads;
      }
    });
    CHECK(std::count(met.begin(), met.end(), 1) == threads);
  }
}

TEST_CASE("a call wakes every sleeping thread it needs") {
  // The caller wakes only the first sleeping worker, and each woken worker
  // wakes the next. Let every worker fall asleep first, then hand out a job
  // whose chunks can only finish if all of them are running at once: a
  // broken chain leaves the job one thread short, and every item times out.
  using clock = std::chrono::steady_clock;
  for (const int threads : {2, 3, 8}) {
    CAPTURE(threads);
    ThreadPool pool(threads);
    for (int round = 0; round < 3; ++round) {
      CAPTURE(round);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));  // far past the spin
      std::atomic<int> arrived{0};
      std::vector<char> met(static_cast<std::size_t>(threads), 0);
      pool.parallel_for(threads, [&](Index begin, Index end) {
        for (Index i = begin; i < end; ++i) {
          arrived.fetch_add(1);
          const auto deadline = clock::now() + std::chrono::seconds(10);
          while (arrived.load() < threads && clock::now() < deadline) std::this_thread::yield();
          met[static_cast<std::size_t>(i)] = arrived.load() >= threads;
        }
      });
      CHECK(std::count(met.begin(), met.end(), 1) == threads);
    }
  }
}

TEST_CASE("an exception in one chunk propagates after every other chunk finishes") {
  ThreadPool pool(4);
  for (Index thrower = 0; thrower < 4; ++thrower) {
    CAPTURE(thrower);
    std::atomic<int> finished{0};
    const auto body = [&](Index begin, Index) {
      if (begin == thrower) throw std::runtime_error("chunk " + std::to_string(begin));
      // Slow enough that a parallel_for returning early is caught below.
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      finished.fetch_add(1);
    };
    const std::string expected = "chunk " + std::to_string(thrower);
    CHECK_THROWS_WITH_AS(pool.parallel_for(4, body), expected.c_str(), std::runtime_error);
    CHECK(finished.load() == 3);
  }
  // Nothing is left locked or half-finished.
  check_cover(pool, 1000, 1);
}

TEST_CASE("when several chunks throw, exactly one exception is rethrown") {
  ThreadPool pool(4);
  std::string what;
  try {
    pool.parallel_for(4, [](Index begin, Index) {
      throw std::runtime_error("chunk " + std::to_string(begin));
    });
  } catch (const std::runtime_error& e) {
    what = e.what();
  }
  const std::set<std::string> any{"chunk 0", "chunk 1", "chunk 2", "chunk 3"};
  CHECK(any.count(what) == 1);
  check_cover(pool, 1000, 1);
}

TEST_CASE("dropped exceptions are destroyed before parallel_for returns") {
  // A dropped exception is destroyed on the thread that ran its chunk. As
  // soon as parallel_for returns the caller may use, or free, whatever that
  // destructor touches, so it has to be over by then. Here it marks a slot
  // in a buffer the caller checks and frees right away: a destructor that
  // runs late leaves a slot unmarked, and is a data race under TSan and a
  // use-after-free under ASan.
  using clock = std::chrono::steady_clock;
  constexpr int threads = 4;
  const std::thread::id caller = std::this_thread::get_id();
  struct Marker {
    int* slot;
    std::thread::id caller;
    ~Marker() {
      // Slow off the calling thread, so that a destructor that starts after
      // its chunk was counted as finished is still running when the caller
      // looks.
      if (std::this_thread::get_id() != caller) {
        std::this_thread::sleep_for(std::chrono::microseconds(500));
      }
      *slot = 1;
    }
  };

  ThreadPool pool(threads);
  int unmarked = 0;
  for (int call = 0; call < 100; ++call) {
    auto destroyed = std::make_unique<int[]>(threads);  // all 0
    std::atomic<int> arrived{0};
    try {
      pool.parallel_for(threads, [&](Index begin, Index) {
        // Each chunk waits for the others to start, so each runs on its own
        // thread and at least two of the exceptions are dropped on workers.
        arrived.fetch_add(1);
        const auto deadline = clock::now() + std::chrono::seconds(10);
        while (arrived.load() < threads && clock::now() < deadline) std::this_thread::yield();
        throw Marker{&destroyed[static_cast<std::size_t>(begin)], caller};
      });
    } catch (const Marker&) {
      // The rethrown one is destroyed here, on the calling thread.
    }
    unmarked +=
        static_cast<int>(threads - std::count(destroyed.get(), destroyed.get() + threads, 1));
    destroyed.reset();
  }
  CHECK(unmarked == 0);
}

TEST_CASE("exceptions of any type propagate, inline or not") {
  ThreadPool pool(4);
  CHECK_THROWS_AS(pool.parallel_for(4, [](Index, Index) { throw 42; }), int);
  CHECK_THROWS_AS(pool.parallel_for(10, 1000, [](Index, Index) { throw 42; }), int);
  ThreadPool one(1);
  CHECK_THROWS_AS(one.parallel_for(100, [](Index, Index) { throw std::out_of_range("x"); }),
                  std::out_of_range);
  check_cover(pool, 100, 1);
  check_cover(one, 100, 1);
}

TEST_CASE("nested parallel_for is rejected") {
  ThreadPool pool(4);
  ThreadPool other(2);
  const auto noop = [](Index, Index) {};

  // From every chunk, including the one on the calling thread.
  CHECK_THROWS_AS(pool.parallel_for(4, [&](Index, Index) { pool.parallel_for(4, noop); }),
                  std::logic_error);
  // From an inline run, so the rule does not depend on n.
  CHECK_THROWS_AS(pool.parallel_for(2, 1000, [&](Index, Index) { pool.parallel_for(4, noop); }),
                  std::logic_error);
  // Into a different pool, which would oversubscribe the machine.
  CHECK_THROWS_AS(pool.parallel_for(4, [&](Index, Index) { other.parallel_for(4, noop); }),
                  std::logic_error);
  // Even when the inner call would do nothing.
  CHECK_THROWS_AS(pool.parallel_for(4, [&](Index, Index) { pool.parallel_for(0, noop); }),
                  std::logic_error);

  // The rejection leaves both pools usable from the top level.
  check_cover(pool, 1000, 1);
  check_cover(other, 1000, 1);
}

TEST_CASE("parallel_for from two threads at once is rejected") {
  ThreadPool pool(2);
  bool rejected = false;
  pool.parallel_for(2, [&](Index begin, Index) {
    if (begin != 0) return;
    // This pool's job is in flight for as long as this chunk runs, and the
    // new thread is not inside a body, so only the in-flight check can stop it.
    std::thread intruder([&] {
      try {
        pool.parallel_for(2, [](Index, Index) {});
      } catch (const std::logic_error&) {
        rejected = true;
      }
    });
    intruder.join();
  });
  CHECK(rejected);
  check_cover(pool, 1000, 1);
}

TEST_CASE("parallel_for rejects a negative n") {
  ThreadPool pool(2);
  int calls = 0;
  CHECK_THROWS_AS(pool.parallel_for(-1, [&](Index, Index) { ++calls; }), std::invalid_argument);
  CHECK(calls == 0);
}

TEST_CASE("pools can be created and destroyed back to back") {
  // Destroying a pool whose workers may not have reached their first wait yet.
  for (int i = 0; i < 50; ++i) {
    ThreadPool pool(8);
  }
  for (int i = 0; i < 50; ++i) {
    ThreadPool pool(3);
    check_cover(pool, 30, 1);
  }
}

TEST_CASE("back-to-back calls see the caller's writes and return theirs") {
  // The caller writes inputs before each call and reads outputs after it; a
  // missing happens-before edge on either side is a TSan report here. The
  // 32-thread pool oversubscribes the machine, so workers routinely wake up
  // after the call they were woken for has finished.
  for (const int threads : {2, 8, 32}) {
    CAPTURE(threads);
    ThreadPool pool(threads);
    constexpr Index n = 64;
    std::vector<int> in(n);
    std::vector<int> out(n);
    int mismatches = 0;
    for (int iter = 0; iter < 2000; ++iter) {
      for (Index i = 0; i < n; ++i) in[static_cast<std::size_t>(i)] = iter + static_cast<int>(i);
      pool.parallel_for(n, [&](Index begin, Index end) {
        for (Index i = begin; i < end; ++i) {
          out[static_cast<std::size_t>(i)] = 2 * in[static_cast<std::size_t>(i)];
        }
      });
      for (Index i = 0; i < n; ++i) {
        if (out[static_cast<std::size_t>(i)] != 2 * (iter + static_cast<int>(i))) ++mismatches;
      }
    }
    CHECK(mismatches == 0);
  }
}

TEST_CASE("no wakeup is lost while threads are falling asleep") {
  // Idle threads spin for a while and then sleep. Sweeping both the gap
  // between calls and the length of a chunk across that window hands out
  // work, and finishes it, at every point of the spin-to-sleep transition.
  // A lost wakeup hangs here (ctest times out) instead of passing.
  using clock = std::chrono::steady_clock;
  const auto busy_wait = [](std::chrono::microseconds us) {
    const auto until = clock::now() + us;
    while (clock::now() < until) {
    }
  };
  for (const int threads : {2, 8}) {
    CAPTURE(threads);
    ThreadPool pool(threads);
    std::vector<int> hits(static_cast<std::size_t>(threads));
    int wrong = 0;
    for (int iter = 0; iter < 400; ++iter) {
      busy_wait(std::chrono::microseconds(iter % 200));
      const auto chunk_time = std::chrono::microseconds((iter * 37) % 200);
      std::fill(hits.begin(), hits.end(), 0);
      pool.parallel_for(threads, [&](Index begin, Index end) {
        // Only the last chunk is slow, so the caller is left waiting on it.
        if (end == threads) busy_wait(chunk_time);
        for (Index i = begin; i < end; ++i) ++hits[static_cast<std::size_t>(i)];
      });
      wrong += static_cast<int>(threads - std::count(hits.begin(), hits.end(), 1));
    }
    CHECK(wrong == 0);
  }
}

TEST_CASE("an idle pool burns no CPU") {
  // A waiting worker spins for 50 us and then sleeps. So once the calls
  // stop, the process's CPU time must stop growing:
  // workers still spinning would add a whole core each, every second. The
  // check allows 10 ms of settling, far more than the spin, and then a
  // quarter of one core over the window, for whatever else the process (or
  // a sanitizer runtime) does meanwhile.
  using namespace std::chrono_literals;
  const auto window = 200ms;
  const auto cpu_during_idle = [&](const std::function<void()>& work) {
    work();
    std::this_thread::sleep_for(10ms);
    const auto start = process_cpu_time();
    std::this_thread::sleep_for(window);
    return process_cpu_time() - start;
  };
  const auto allowed = window / 4;
  std::vector<int> sink(1000);
  const Body touch = [&](Index begin, Index end) {
    for (Index i = begin; i < end; ++i) ++sink[static_cast<std::size_t>(i)];
  };

  for (const int threads : {4, 8}) {
    CAPTURE(threads);
    ThreadPool pool(threads);
    const auto cpu = cpu_during_idle([&] {
      for (int call = 0; call < 200; ++call) pool.parallel_for(1000, 1, touch);
    });
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu).count();
    CAPTURE(cpu_ms);
    CHECK(cpu < allowed);
  }

  KernelThreads guard(4);
  const auto cpu = cpu_during_idle([&] {
    for (int call = 0; call < 200; ++call) kernel_parallel_for(1000, 1, touch);
  });
  const double cpu_ms = std::chrono::duration<double, std::milli>(cpu).count();
  CAPTURE(cpu_ms);
  CHECK(cpu < allowed);
}

// ---------------------------------------------------------------------------
// Arena
// ---------------------------------------------------------------------------

TEST_CASE("Arena honours alignment") {
  Arena arena(1 << 16);
  for (const std::size_t align : {1, 2, 8, 16, 64, 256, 4096, 8192}) {
    CAPTURE(align);
    arena.allocate(1, 1);  // knock the offset off every natural boundary
    void* p = arena.allocate(24, align);
    CHECK(aligned(p, align));
    std::memset(p, 0xab, 24);
  }

  arena.allocate(3, 1);
  float* f = arena.allocate_array<float>(10);
  CHECK(aligned(f, 64));  // allocate_array never asks for less than a cache line
  for (int i = 0; i < 10; ++i) f[i] = static_cast<float>(i);

  struct alignas(128) Wide {
    float v[32];
  };
  arena.allocate(3, 1);
  Wide* w = arena.allocate_array<Wide>(2);
  CHECK(aligned(w, 128));
  w[1].v[31] = 1.0f;
}

TEST_CASE("Arena::allocate_array throws instead of wrapping count * sizeof(T)") {
  Arena arena(1024);
  arena.allocate(10, 1);
  // 2^62 floats is 2^64 bytes, which wraps to a 0-byte request that would
  // succeed; one more float wraps to 4 bytes.
  CHECK_THROWS_AS(arena.allocate_array<float>(std::size_t{1} << 62), std::bad_alloc);
  CHECK_THROWS_AS(arena.allocate_array<float>((std::size_t{1} << 62) + 1), std::bad_alloc);
  struct Wide {
    float v[32];
  };
  constexpr std::size_t max_wide = std::numeric_limits<std::size_t>::max() / sizeof(Wide);
  CHECK_THROWS_AS(arena.allocate_array<Wide>(max_wide + 1), std::bad_alloc);
  // The largest count that does not wrap reaches the arena's own check.
  CHECK_THROWS_AS(arena.allocate_array<Wide>(max_wide), std::bad_alloc);
  CHECK(arena.used() == 10);
  CHECK_NOTHROW(arena.allocate_array<float>(0));
}

TEST_CASE("Arena allocations do not overlap") {
  Arena arena(4096);
  auto* a = static_cast<unsigned char*>(arena.allocate(100, 1));
  auto* b = static_cast<unsigned char*>(arena.allocate(100, 64));
  auto* c = static_cast<unsigned char*>(arena.allocate(7, 1));
  CHECK(b >= a + 100);
  CHECK(c >= b + 100);
  std::memset(a, 1, 100);
  std::memset(b, 2, 100);
  std::memset(c, 3, 7);
  CHECK(std::count(a, a + 100, 1) == 100);
  CHECK(std::count(b, b + 100, 2) == 100);
}

TEST_CASE("Arena used() counts padding and is the same for the same requests") {
  // The block is aligned well past a cache line, so a fresh arena pads
  // nothing for align <= 64 and used() depends only on the requests. That is
  // what makes peak() a stable number to size the model's arena from.
  for (int run = 0; run < 3; ++run) {
    Arena arena(4096);
    CHECK(arena.used() == 0);
    arena.allocate(1, 1);
    CHECK(arena.used() == 1);
    arena.allocate(4, 64);
    CHECK(arena.used() == 68);  // 63 bytes of padding, then 4
    arena.allocate(0, 1);
    CHECK(arena.used() == 68);
    arena.allocate_array<float>(1);
    CHECK(arena.used() == 132);
  }
  // Up to a page, in fact, whatever alignment malloc happens to give.
  for (int run = 0; run < 3; ++run) {
    Arena arena(3 * 4096);
    arena.allocate(1, 4096);
    CHECK(arena.used() == 1);
  }
}

TEST_CASE("Arena throws bad_alloc on exhaustion and is unchanged by the failure") {
  Arena arena(256);
  CHECK(arena.capacity() == 256);
  arena.allocate(200, 1);
  CHECK_THROWS_AS(arena.allocate(57, 1), std::bad_alloc);
  CHECK(arena.used() == 200);
  CHECK(arena.peak() == 200);

  arena.allocate(56, 1);  // exactly full
  CHECK(arena.used() == 256);
  CHECK_THROWS_AS(arena.allocate(1, 1), std::bad_alloc);
  CHECK_NOTHROW(arena.allocate(0, 1));

  // Padding alone can exhaust it.
  Arena small(100);
  small.allocate(1, 1);
  CHECK_THROWS_AS(small.allocate(1, 128), std::bad_alloc);
  CHECK(small.used() == 1);

  // No request is large enough to wrap the bounds arithmetic.
  constexpr std::size_t huge = std::numeric_limits<std::size_t>::max();
  CHECK_THROWS_AS(small.allocate(huge, 1), std::bad_alloc);
  CHECK_THROWS_AS(small.allocate(huge - 8, 64), std::bad_alloc);
  CHECK_THROWS_AS(small.allocate(1, std::size_t{1} << 63), std::bad_alloc);
  CHECK(small.used() == 1);

  Arena none(0);
  CHECK_NOTHROW(none.allocate(0, 1));
  CHECK_THROWS_AS(none.allocate(1, 1), std::bad_alloc);

  // The message says how short the arena was.
  Arena sized(1000);
  sized.allocate(600);
  try {
    sized.allocate(500);
    FAIL_CHECK("expected std::bad_alloc");
  } catch (const std::bad_alloc& e) {
    const std::string what = e.what();
    CAPTURE(what);
    CHECK(what.find("500 bytes") != std::string::npos);
    CHECK(what.find("400 of 1000 bytes free") != std::string::npos);
  }
}

TEST_CASE("Arena rejects an alignment that is not a power of two") {
  Arena arena(1024);
  for (const std::size_t align : {0, 3, 6, 48, 100}) {
    CAPTURE(align);
    CHECK_THROWS_AS(arena.allocate(8, align), std::invalid_argument);
  }
  CHECK(arena.used() == 0);
}

TEST_CASE("reset() rewinds used() but peak() keeps the high-water mark") {
  Arena arena(4096);
  CHECK(arena.peak() == 0);
  void* first = arena.allocate(1000);
  arena.allocate(500);
  const std::size_t high = arena.used();
  CHECK(high >= 1500);
  CHECK(arena.peak() == high);

  arena.reset();
  CHECK(arena.used() == 0);
  CHECK(arena.peak() == high);

  // The same requests get the same memory back.
  CHECK(arena.allocate(1000) == first);
  CHECK(arena.peak() == high);  // a smaller pass does not lower it

  arena.reset();
  arena.allocate(3000);
  CHECK(arena.peak() == 3000);
  CHECK_THROWS_AS(arena.allocate(2000), std::bad_alloc);
  CHECK(arena.peak() == 3000);  // a failed request does not raise it
  arena.reset();
  arena.reset();
  CHECK(arena.used() == 0);
  CHECK(arena.peak() == 3000);
}

TEST_CASE("Arena::tensor returns writable non-owning views") {
  Arena arena(1 << 16);
  Tensor t = arena.tensor({4, 8});
  CHECK(t.shape() == Shape{4, 8});
  CHECK(t.dtype() == DType::F32);
  CHECK(t.is_contiguous());
  CHECK_FALSE(t.owns_data());
  CHECK(aligned(t.data(), 64));
  CHECK(arena.used() == 4 * 8 * sizeof(float));

  float* p = t.f32();
  for (int i = 0; i < 32; ++i) p[i] = static_cast<float>(i);

  // A second tensor is fresh memory, not an alias of the first.
  Tensor u = arena.tensor({3, 5});
  CHECK(aligned(u.data(), 64));
  for (int i = 0; i < 15; ++i) u.f32()[i] = -1.0f;
  for (int i = 0; i < 32; ++i) CHECK(p[i] == static_cast<float>(i));

  // ops:: take contiguous F32 views and slice them with row/reshape/slice;
  // all of that has to work over arena memory. (This test links only
  // minillm_runtime, so it cannot call ops:: itself.)
  CHECK(t.row(2).f32()[0] == 16.0f);
  CHECK(t.reshape({32}).f32()[31] == 31.0f);
  CHECK(t.reshape({2, 2, 8}).is_contiguous());
  CHECK(t.slice(0, 1, 2).f32()[0] == 8.0f);

  Tensor half = arena.tensor({3}, DType::BF16);
  CHECK(half.dtype() == DType::BF16);
  half.data_as<std::uint16_t>()[2] = 0x3f80;

  Tensor empty = arena.tensor({0, 8});
  CHECK(empty.numel() == 0);
  CHECK(empty.shape() == Shape{0, 8});
}

TEST_CASE("Arena::tensor validates before allocating") {
  Arena arena(1024);
  arena.allocate(10, 1);
  CHECK_THROWS_AS(arena.tensor({4}, DType::Q8_0), std::invalid_argument);
  CHECK_THROWS_AS(arena.tensor(Shape{}), std::invalid_argument);
  CHECK_THROWS_AS(arena.tensor({2, -1}), std::invalid_argument);
  CHECK_THROWS_AS(arena.tensor({16, 16}), std::bad_alloc);  // 1024 bytes, 1014 free
  // 2^62 elements: a valid shape, but numel * 4 would wrap a 64-bit size_t to 0.
  CHECK_THROWS_AS(arena.tensor({std::int64_t{1} << 30, std::int64_t{1} << 30, 4}),
                  std::bad_alloc);
  // 2^124 elements: the element count overflows int64, so no tensor can have
  // this shape (Tensor::wrap rejects it too).
  CHECK_THROWS_AS(arena.tensor({std::int64_t{1} << 62, std::int64_t{1} << 62}),
                  std::invalid_argument);
  // Empty, but dimension 0 would need a stride of 2^124: rejected before the
  // arena moves, not by Tensor::wrap after it has.
  CHECK_THROWS_AS(arena.tensor({0, std::int64_t{1} << 62, std::int64_t{1} << 62}),
                  std::invalid_argument);
  CHECK(arena.used() == 10);
  // A zero extent outside the large ones leaves every stride small: a valid
  // empty tensor, however large the other extents are.
  CHECK(arena.tensor({std::int64_t{1} << 62, 0, std::int64_t{1} << 62}).numel() == 0);
}

// Only an ASan build poisons arena memory, so only an ASan build has this
// test. Elsewhere there is nothing to check, and so nothing skipped.
#ifdef MINILLM_TEST_ASAN
TEST_CASE("Arena memory is poisoned after reset (ASan builds)") {
  Arena arena(4096);
  auto* p = static_cast<unsigned char*>(arena.allocate(64));
  CHECK_FALSE(__asan_address_is_poisoned(p));
  CHECK_FALSE(__asan_address_is_poisoned(p + 63));
  CHECK(__asan_address_is_poisoned(p + 64));  // never handed out

  Tensor t = arena.tensor({16});
  t.f32()[15] = 1.0f;
  CHECK_FALSE(__asan_address_is_poisoned(&t.f32()[15]));

  arena.reset();
  // Any read through `p` or `t` from here on is an ASan report.
  CHECK(__asan_address_is_poisoned(p));
  CHECK(__asan_address_is_poisoned(&t.f32()[15]));

  // And the next pass gets the memory back, addressable again.
  auto* q = static_cast<unsigned char*>(arena.allocate(64));
  CHECK(q == p);
  CHECK_FALSE(__asan_address_is_poisoned(q));
}
#endif

TEST_CASE("arena scratch can be filled by a thread pool") {
  // How the engine uses the two together: one arena per pass, owned by the
  // calling thread; the pool writes disjoint rows of a view into it.
  Arena arena(1 << 20);
  ThreadPool pool(4);
  for (int pass = 0; pass < 3; ++pass) {
    arena.reset();
    Tensor t = arena.tensor({256, 64});
    float* p = t.f32();
    pool.parallel_for(256, 16, [&](Index begin, Index end) {
      for (Index r = begin; r < end; ++r) {
        for (Index c = 0; c < 64; ++c) p[r * 64 + c] = static_cast<float>(pass * 100000 + r * 64 + c);
      }
    });
    int wrong = 0;
    for (Index i = 0; i < 256 * 64; ++i) {
      if (p[i] != static_cast<float>(pass * 100000 + i)) ++wrong;
    }
    CHECK(wrong == 0);
  }
  CHECK(arena.peak() == 256 * 64 * sizeof(float));
}

// ---------------------------------------------------------------------------
// kernel_parallel_for and the process-wide pool
// ---------------------------------------------------------------------------

TEST_CASE("set_num_threads resolves n <= 0 to the default, and num_threads reports it") {
  const int def = expected_default_threads();
  KernelThreads guard(3);
  CHECK(num_threads() == 3);
  set_num_threads(0);
  CHECK(num_threads() == def);
  set_num_threads(1);
  CHECK(num_threads() == 1);
  set_num_threads(-7);
  CHECK(num_threads() == def);
  set_num_threads(5);
  CHECK(num_threads() == 5);
}

TEST_CASE("kernel_parallel_for covers [0, n) exactly once at every thread count") {
  const std::thread::id caller = std::this_thread::get_id();
  for (const int threads : kThreadCounts) {
    KernelThreads guard(threads);
    for (const Index n : {1, 2, 7, 100, 1000, 49152}) {
      for (const Index grain : {1, 3, 64}) {
        const Cover cover = check_kernel_cover(n, grain);
        if (threads == 1 || n / grain <= 1) {
          // One thread, or less than two grains of work: inline, one chunk.
          CHECK(cover.chunks == 1);
          CHECK(cover.threads == std::set<std::thread::id>{caller});
        } else {
          CHECK(cover.chunks > 1);
        }
      }
    }
  }
}

TEST_CASE("set_num_threads starts and stops the pool's threads") {
  std::thread([] {}).join();  // start any thread a sanitizer runtime starts lazily
  KernelThreads guard(1);
  const int base = settled_thread_count();
  if (base < 0) {
    note_skip("runtime: no OS thread count on this platform; kernel pool size checks skipped");
    MESSAGE("SKIP: no OS thread count on this platform");
    return;
  }
  for (const int threads : {4, 2, 10, 1, 3}) {
    CAPTURE(threads);
    set_num_threads(threads);
    check_kernel_cover(1000, 1);
    CHECK(settled_thread_count() == base + threads - 1);
  }
}

TEST_CASE("kernel_parallel_for rejects a negative n and does nothing for n = 0") {
  KernelThreads guard(4);
  int calls = 0;
  CHECK_THROWS_AS(kernel_parallel_for(-1, 1, [&](Index, Index) { ++calls; }),
                  std::invalid_argument);
  kernel_parallel_for(0, 1, [&](Index, Index) { ++calls; });
  CHECK(calls == 0);
}

TEST_CASE("an exception leaves kernel_parallel_for usable, pooled or inline") {
  for (const int threads : {1, 4}) {
    CAPTURE(threads);
    KernelThreads guard(threads);
    CHECK_THROWS_AS(kernel_parallel_for(100, 1, [](Index, Index) { throw std::out_of_range("x"); }),
                    std::out_of_range);
    CHECK_THROWS_AS(kernel_parallel_for(5, 100, [](Index, Index) { throw 42; }), int);
    // The pool is not left held: the next call can use it again.
    check_kernel_cover(1000, 1);
  }
}

TEST_CASE("a nested kernel_parallel_for runs inline on the thread that called it") {
  KernelThreads guard(4);
  constexpr Index outer = 16;
  constexpr Index inner = 1000;
  const auto un = static_cast<std::size_t>(outer);
  std::vector<int> chunks(un, 0);
  std::vector<Index> covered(un, 0);
  std::vector<int> elsewhere(un, 0);
  std::vector<int> still_in_body(un, 0);
  ThreadPool other(2);
  kernel_parallel_for(outer, 1, [&](Index begin, Index end) {
    for (Index i = begin; i < end; ++i) {
      const auto slot = static_cast<std::size_t>(i);
      const std::thread::id me = std::this_thread::get_id();
      kernel_parallel_for(inner, 1, [&](Index b, Index e) {
        ++chunks[slot];
        covered[slot] += e - b;
        if (std::this_thread::get_id() != me) ++elsewhere[slot];
      });
      // The inline run must not have cleared "inside a body" on its way out.
      try {
        other.parallel_for(4, [](Index, Index) {});
      } catch (const std::logic_error&) {
        still_in_body[slot] = 1;
      }
    }
  });
  for (std::size_t i = 0; i < un; ++i) {
    CAPTURE(i);
    CHECK(chunks[i] == 1);
    CHECK(covered[i] == inner);
    CHECK(elsewhere[i] == 0);
    CHECK(still_in_body[i] == 1);
  }
}

TEST_CASE("kernel_parallel_for inside a ThreadPool body runs inline instead of throwing") {
  // ops:: called from a caller's own pool: the kernel must not nest into the
  // process-wide one.
  KernelThreads guard(4);
  ThreadPool pool(3);
  std::vector<int> chunks(3, 0);
  std::vector<Index> covered(3, 0);
  pool.parallel_for(3, [&](Index begin, Index end) {
    for (Index i = begin; i < end; ++i) {
      const auto slot = static_cast<std::size_t>(i);
      kernel_parallel_for(500, 1, [&](Index b, Index e) {
        ++chunks[slot];
        covered[slot] += e - b;
      });
    }
  });
  CHECK(chunks == std::vector<int>{1, 1, 1});
  CHECK(covered == std::vector<Index>{500, 500, 500});
}

TEST_CASE("while the pool is busy, another thread's kernel_parallel_for runs inline") {
  KernelThreads guard(4);
  Index intruder_chunks = 0;
  Index intruder_items = 0;
  bool intruder_inline = true;
  // The pool is held for as long as any chunk of this call runs.
  kernel_parallel_for(4, 1, [&](Index begin, Index) {
    if (begin != 0) return;
    std::thread intruder([&] {
      const std::thread::id me = std::this_thread::get_id();
      kernel_parallel_for(1000, 1, [&](Index b, Index e) {
        ++intruder_chunks;
        intruder_items += e - b;
        if (std::this_thread::get_id() != me) intruder_inline = false;
      });
    });
    intruder.join();
  });
  CHECK(intruder_chunks == 1);
  CHECK(intruder_items == 1000);
  CHECK(intruder_inline);
}

TEST_CASE("concurrent kernel_parallel_for callers each cover their range exactly once") {
  // Whichever caller gets the pool uses it; the others run inline. Either
  // way every caller's range is covered once per call.
  for (const int threads : {2, 4, 10}) {
    CAPTURE(threads);
    KernelThreads guard(threads);
    constexpr int callers = 6;
    constexpr int calls = 200;
    constexpr Index n = 777;
    std::vector<std::vector<int>> hits(callers, std::vector<int>(static_cast<std::size_t>(n), 0));
    std::vector<std::thread> pool_users;
    for (int c = 0; c < callers; ++c) {
      pool_users.emplace_back([&hits, c] {
        std::vector<int>& mine = hits[static_cast<std::size_t>(c)];
        for (int call = 0; call < calls; ++call) {
          kernel_parallel_for(n, 1, [&](Index begin, Index end) {
            for (Index i = begin; i < end; ++i) ++mine[static_cast<std::size_t>(i)];
          });
        }
      });
    }
    for (std::thread& t : pool_users) t.join();
    int wrong = 0;
    for (const std::vector<int>& h : hits) {
      wrong += static_cast<int>(std::count_if(h.begin(), h.end(), [](int x) { return x != calls; }));
    }
    CHECK(wrong == 0);
  }
}

TEST_CASE("set_num_threads waits for the running kernel, then resizes the pool") {
  using clock = std::chrono::steady_clock;
  KernelThreads guard(4);
  std::atomic<bool> resizing{false};
  bool kernel_finished = false;  // plain: only the pool's own ordering protects it
  bool resizer_saw_finished = false;
  std::thread resizer;
  kernel_parallel_for(4, 1, [&](Index begin, Index) {
    if (begin != 0) return;
    resizer = std::thread([&] {
      resizing.store(true);
      set_num_threads(3);
      resizer_saw_finished = kernel_finished;
    });
    const auto deadline = clock::now() + std::chrono::seconds(10);
    while (!resizing.load() && clock::now() < deadline) std::this_thread::yield();
    // Long enough for a set_num_threads that does not wait to have returned.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    kernel_finished = true;
  });
  resizer.join();
  CHECK(resizer_saw_finished);
  CHECK(num_threads() == 3);
  check_kernel_cover(1000, 1);
}

TEST_CASE("set_num_threads racing running kernels leaves every call whole") {
  KernelThreads guard(2);
  constexpr int callers = 3;
  constexpr Index n = 512;
  std::atomic<bool> stop{false};
  std::vector<std::vector<int>> hits(callers, std::vector<int>(static_cast<std::size_t>(n), 0));
  std::vector<int> calls(callers, 0);
  std::vector<std::thread> pool_users;
  for (int c = 0; c < callers; ++c) {
    pool_users.emplace_back([&, c] {
      std::vector<int>& mine = hits[static_cast<std::size_t>(c)];
      while (!stop.load()) {
        kernel_parallel_for(n, 1, [&](Index begin, Index end) {
          for (Index i = begin; i < end; ++i) ++mine[static_cast<std::size_t>(i)];
        });
        ++calls[static_cast<std::size_t>(c)];
      }
    });
  }
  for (int i = 0; i < 100; ++i) set_num_threads(kThreadCounts[i % 5]);
  stop.store(true);
  for (std::thread& t : pool_users) t.join();
  for (int c = 0; c < callers; ++c) {
    CAPTURE(c);
    const std::vector<int>& h = hits[static_cast<std::size_t>(c)];
    const int expected = calls[static_cast<std::size_t>(c)];
    CHECK(std::count(h.begin(), h.end(), expected) == n);
  }
}

TEST_CASE("set_num_threads from inside a kernel body is rejected") {
  // The kernel holds the pool while it waits for the body, so the call could
  // never return.
  KernelThreads guard(4);
  std::vector<int> rejected(4, 0);
  kernel_parallel_for(4, 1, [&](Index begin, Index end) {
    for (Index i = begin; i < end; ++i) {
      try {
        set_num_threads(2);
      } catch (const std::logic_error&) {
        rejected[static_cast<std::size_t>(i)] = 1;
      }
    }
  });
  CHECK(rejected == std::vector<int>{1, 1, 1, 1});
  int inline_rejected = 0;
  kernel_parallel_for(1, 1, [&](Index, Index) {
    try {
      set_num_threads(2);
    } catch (const std::logic_error&) {
      inline_rejected = 1;
    }
  });
  CHECK(inline_rejected == 1);
  CHECK(num_threads() == 4);
}
