// minillm benchmarks — owned by workflow 10-bench-ci.
//
// STUB. Deliberately no benchmarking framework: timing a kernel is a loop and
// a clock, and the dependency list in docs/ARCHITECTURE.md §6 is closed.
//
// What this should eventually measure (all on the `release` preset, never on
// a sanitizer build):
//   * each op in ops.h at the model's real shapes, ops:: vs ref::
//   * prefill tokens/s and decode tokens/s for the whole model
//   * peak resident memory
// and print one machine-readable line per measurement so CI can track drift.

#include <cstdio>

int main() {
  std::fprintf(stderr, "minillm_bench: not implemented yet (workflow 10-bench-ci)\n");
  return 1;
}
