// Checks that a kernel's output does not depend on the kernel thread count.
//
// Header-only: only the op tests include it, and they link the runtime that
// set_num_threads lives in; the shared test library does not.
//
// Every threaded op in minillm::ops promises output that is bitwise identical
// at any thread count. That keeps oracle comparisons valid at every
// --threads, and it makes a race show up as a test failure rather than as
// noise in the last bits.

#ifndef MINILLM_TESTS_COMMON_THREAD_INVARIANT_H
#define MINILLM_TESTS_COMMON_THREAD_INVARIANT_H

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>

#include "minillm/runtime.h"
#include "minillm/tensor.h"

namespace minillm::test {

/// The kernel thread counts the threaded path is checked at. 10 is the
/// development machine's hardware_concurrency(); on a smaller machine it
/// oversubscribes, which exercises the split just as well.
inline constexpr int kThreadCounts[] = {1, 2, 3, 4, 10};

/// Calls run() at threads = 1, then at every other count in kThreadCounts,
/// and checks that each result is bitwise identical to the 1-thread one,
/// which it returns. Restores the thread count, pass or fail.
inline Tensor check_thread_invariant(const std::string& label,
                                     const std::function<Tensor()>& run) {
  struct Restore {
    int threads;
    ~Restore() { set_num_threads(threads); }
  } restore{num_threads()};
  const auto exact = [](float v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(v));
    return std::string(buf);
  };
  set_num_threads(1);
  Tensor single = run();
  for (const int threads : kThreadCounts) {
    if (threads == 1) continue;
    set_num_threads(threads);
    const Tensor multi = run();
    REQUIRE(multi.shape() == single.shape());
    const float* a = multi.f32();
    const float* b = single.f32();
    for (std::int64_t i = 0; i < single.numel(); ++i) {
      if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
        FAIL_CHECK(label << ": threads = " << threads << " differs from threads = 1 at index "
                         << i << ": " << exact(a[i]) << " vs " << exact(b[i]));
        break;
      }
    }
  }
  return single;
}

}  // namespace minillm::test

#endif  // MINILLM_TESTS_COMMON_THREAD_INVARIANT_H
