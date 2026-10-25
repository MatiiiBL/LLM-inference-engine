// Numeric comparison with a useful failure message.
//
// A bare "values differ" tells you nothing; these report the max absolute
// error, the max relative error, and the first mismatching index with both
// values, so a failure localizes the bug without another printf run.

#ifndef MINILLM_TESTS_COMMON_ALLCLOSE_H
#define MINILLM_TESTS_COMMON_ALLCLOSE_H

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "common/tolerances.h"
#include "minillm/tensor.h"

namespace minillm::test {

/// The outcome of a comparison. Always computed in full (the whole array is
/// scanned even after the first mismatch) so the error summary is meaningful.
struct CloseResult {
  bool ok = false;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  std::int64_t first_mismatch = -1;  ///< -1 when ok
  double first_actual = 0.0;
  double first_expected = 0.0;
  std::int64_t n_mismatches = 0;
  std::int64_t n = 0;
  /// Multi-line summary, empty when ok.
  std::string message;
};

/// Compares element by element: |a - e| <= atol + rtol * |e|.
/// NaN matches NaN; infinities must match in sign.
CloseResult compare(std::span<const float> actual, std::span<const float> expected, Tolerance tol);

/// Tensor overload. Shapes must match exactly; both must be F32. A shape
/// mismatch is reported as a failure, not an exception.
CloseResult compare(const Tensor& actual, const Tensor& expected, Tolerance tol);

/// doctest-reporting assertions. On failure these CHECK(false) with the full
/// summary, so the test name and the message land in the ctest output
/// together. `label` is prepended to the message — use it to say which op and
/// which shape, e.g. "matmul [8, 576] x [576, 576]".
void assert_allclose(std::span<const float> actual, std::span<const float> expected, Tolerance tol,
                     const std::string& label = {});
void assert_allclose(const Tensor& actual, const Tensor& expected, Tolerance tol,
                     const std::string& label = {});

/// Exact equality for token ids / argmax results. Reports the first
/// difference and how many differ.
void assert_equal(std::span<const std::int32_t> actual, std::span<const std::int32_t> expected,
                  const std::string& label = {});

}  // namespace minillm::test

#endif  // MINILLM_TESTS_COMMON_ALLCLOSE_H
