// Workflow 08-runtime owns this file.
//
// PLACEHOLDER. ThreadPool::parallel_for and Arena::allocate are stubs.
//
// What this must cover when it is done:
//   * parallel_for covers [0, n) exactly once with no gaps or overlaps — have
//     each chunk write its indices into a vector and check every slot is 1
//   * it works for n = 0, n = 1, n < size(), n not divisible by size()
//   * grain is respected: a small n runs inline
//   * an exception in a chunk propagates to the caller
//   * a pool of size 1 starts no threads
//   * run the whole suite under the tsan preset, not just asan
//   * Arena: alignment honoured, bad_alloc on exhaustion, reset() rewinds,
//     peak() tracks the high-water mark, tensor() views are usable by ops

#include <doctest/doctest.h>

#include <stdexcept>

#include "common/oracle.h"
#include "minillm/runtime.h"

using namespace minillm;
using namespace minillm::test;

TEST_CASE("a ThreadPool can be constructed before parallel_for exists") {
  ThreadPool pool(1);
  CHECK(pool.size() >= 1);
}

TEST_CASE("an Arena reports the capacity it was asked for") {
  Arena arena(1024);
  CHECK(arena.capacity() == 1024);
  CHECK(arena.used() == 0);
}

TEST_CASE("runtime stubs are linked and announce themselves") {
  MESSAGE("SKIP: the thread pool and arena are stubs — workflow 08-runtime implements them");
  note_skip("runtime: parallel_for and Arena tests pending workflow 08-runtime");

  ThreadPool pool(2);
  CHECK_THROWS_AS(pool.parallel_for(10, [](std::int64_t, std::int64_t) {}), std::logic_error);
  CHECK_THROWS_AS(pool.parallel_for(10, 4, [](std::int64_t, std::int64_t) {}), std::logic_error);

  Arena arena(256);
  CHECK_THROWS_AS(arena.allocate(16), std::logic_error);
  CHECK_THROWS_AS(arena.tensor({2, 2}), std::logic_error);
  CHECK_THROWS_AS(arena.reset(), std::logic_error);
}
