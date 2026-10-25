// Shared main() for every test executable.
//
// Added to each test target's sources (see tests/CMakeLists.txt) rather than
// living in the test-common library, so the linker can never drop it.

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "common/oracle.h"

int main(int argc, char** argv) {
  doctest::Context context;
  context.applyCommandLine(argc, argv);
  const int result = context.run();
  // Make skipped oracle checks visible even on a green run: a suite that
  // silently skipped everything looks exactly like a suite that passed.
  minillm::test::print_skip_summary();
  if (context.shouldExit()) return result;
  return result;
}
