// Locating the PyTorch golden data, and skipping cleanly without it.
//
// The oracle directory is baked in at configure time (MINILLM_ORACLE_DIR) and
// its contents are gitignored, so a fresh clone has none of it. Tests that
// need golden data must therefore skip — loudly, naming the missing file —
// rather than fail. A silently skipped suite is worse than a red one.

#ifndef MINILLM_TESTS_COMMON_ORACLE_H
#define MINILLM_TESTS_COMMON_ORACLE_H

#include <doctest/doctest.h>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "common/npy.h"

namespace minillm::test {

/// The directory the oracle writes into (MINILLM_ORACLE_DIR).
const std::filesystem::path& oracle_dir();

/// The model checkpoint directory (MINILLM_MODELS_DIR).
const std::filesystem::path& models_dir();

/// oracle_dir() / relative.
std::filesystem::path oracle_path(std::string_view relative);

/// True if that file exists under the oracle directory.
bool oracle_has(std::string_view relative);

/// Loads an oracle array. Throws std::runtime_error if it is missing — guard
/// with MINILLM_SKIP_WITHOUT_ORACLE first.
NpyArray oracle_npy(std::string_view relative);

/// Records a skip and prints it, so skips are visible in the test output and
/// summarized at exit.
void note_skip(const std::string& reason);
const std::vector<std::string>& skips();

/// Prints the end-of-run skip summary. Called by the shared doctest main.
void print_skip_summary();

}  // namespace minillm::test

/// Skips the rest of the current test case unless every listed oracle file is
/// present. Use it as the first line of any test that reads golden data:
///
///   TEST_CASE("matmul matches torch") {
///     MINILLM_SKIP_WITHOUT_ORACLE({"ops/matmul_small_x.npy", "ops/matmul_small_out.npy"});
///     ...
///   }
/// Note: the list holds std::string, not std::string_view — callers build
/// paths with `base + "x.npy"`, and views of those temporaries would dangle.
#define MINILLM_SKIP_WITHOUT_ORACLE(...)                                      \
  do {                                                                        \
    const std::vector<std::string> minillm_needed__ __VA_ARGS__;               \
    std::string minillm_missing__;                                            \
    for (const std::string& minillm_rel__ : minillm_needed__) {               \
      if (!minillm::test::oracle_has(minillm_rel__)) {                         \
        if (!minillm_missing__.empty()) minillm_missing__ += ", ";             \
        minillm_missing__ += minillm_rel__;                                    \
      }                                                                       \
    }                                                                         \
    if (!minillm_missing__.empty()) {                                         \
      const std::string minillm_reason__ =                                    \
          "missing oracle data: " + minillm_missing__ +                        \
          " — run `python oracle/generate.py`";                                \
      minillm::test::note_skip(minillm_reason__);                             \
      MESSAGE("SKIP: " << minillm_reason__);                                  \
      return;                                                                 \
    }                                                                         \
  } while (false)

#endif  // MINILLM_TESTS_COMMON_ORACLE_H
