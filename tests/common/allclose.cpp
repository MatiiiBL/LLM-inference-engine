#include "common/allclose.h"

#include <cmath>
#include <doctest/doctest.h>
#include <sstream>

namespace minillm::test {
namespace {

std::string format_summary(const CloseResult& r, const std::string& label, Tolerance tol) {
  std::ostringstream os;
  os.precision(9);
  if (!label.empty()) os << label << ": ";
  os << r.n_mismatches << " of " << r.n << " elements outside tolerance "
     << "(rtol=" << tol.rtol << ", atol=" << tol.atol << ")\n"
     << "  max abs error : " << r.max_abs_error << '\n'
     << "  max rel error : " << r.max_rel_error << '\n'
     << "  first mismatch: index " << r.first_mismatch << " actual " << r.first_actual
     << " expected " << r.first_expected;
  return os.str();
}

}  // namespace

CloseResult compare(std::span<const float> actual, std::span<const float> expected, Tolerance tol) {
  CloseResult r;
  r.n = static_cast<std::int64_t>(expected.size());

  if (actual.size() != expected.size()) {
    std::ostringstream os;
    os << "length mismatch: actual has " << actual.size() << " elements, expected has "
       << expected.size();
    r.message = os.str();
    return r;
  }

  for (std::size_t i = 0; i < expected.size(); ++i) {
    const double a = actual[i];
    const double e = expected[i];

    bool element_ok;
    if (std::isnan(a) || std::isnan(e)) {
      element_ok = std::isnan(a) && std::isnan(e);
    } else if (std::isinf(a) || std::isinf(e)) {
      element_ok = (a == e);
    } else {
      const double abs_err = std::abs(a - e);
      const double rel_err = std::abs(e) > 0.0 ? abs_err / std::abs(e) : 0.0;
      if (abs_err > r.max_abs_error) r.max_abs_error = abs_err;
      if (rel_err > r.max_rel_error) r.max_rel_error = rel_err;
      element_ok = abs_err <= tol.atol + tol.rtol * std::abs(e);
    }

    if (!element_ok) {
      ++r.n_mismatches;
      if (r.first_mismatch < 0) {
        r.first_mismatch = static_cast<std::int64_t>(i);
        r.first_actual = a;
        r.first_expected = e;
      }
    }
  }

  r.ok = (r.n_mismatches == 0);
  if (!r.ok) r.message = format_summary(r, {}, tol);
  return r;
}

CloseResult compare(const Tensor& actual, const Tensor& expected, Tolerance tol) {
  CloseResult r;
  if (actual.dtype() != DType::F32 || expected.dtype() != DType::F32) {
    r.message = "compare: both tensors must be f32, got " + actual.to_string() + " and " +
                expected.to_string();
    return r;
  }
  if (actual.shape() != expected.shape()) {
    r.message = "compare: shape mismatch: actual " + actual.shape().to_string() + " vs expected " +
                expected.shape().to_string();
    return r;
  }
  if (!actual.is_contiguous() || !expected.is_contiguous()) {
    r.message = "compare: both tensors must be contiguous (use copy_contiguous()), got " +
                actual.to_string() + " and " + expected.to_string();
    return r;
  }
  const auto n = static_cast<std::size_t>(expected.numel());
  return compare(std::span<const float>(actual.f32(), n), std::span<const float>(expected.f32(), n),
                 tol);
}

namespace {

/// Reports a finished comparison to doctest.
///
/// Always registers exactly one assertion, including on success: a comparison
/// that ran and passed must be distinguishable from a test that compared
/// nothing at all. Comparing zero elements is therefore a failure — it only
/// happens when a shape is wrong or a loop never executed.
void report(const CloseResult& r, const std::string& label, Tolerance tol) {
  if (r.ok && r.n > 0) {
    CHECK(r.n_mismatches == 0);
    return;
  }
  std::string message;
  if (r.first_mismatch >= 0) {
    message = format_summary(r, label, tol);
  } else if (!r.message.empty()) {
    message = label.empty() ? r.message : label + ": " + r.message;
  } else {
    message = (label.empty() ? std::string("comparison") : label) +
              ": compared zero elements — the assertion did nothing";
  }
  // CHECK rather than REQUIRE: one bad shape in a loop should not hide the
  // rest of the comparisons in the same test case.
  CHECK_MESSAGE(false, message);
}

}  // namespace

void assert_allclose(std::span<const float> actual, std::span<const float> expected, Tolerance tol,
                     const std::string& label) {
  report(compare(actual, expected, tol), label, tol);
}

void assert_allclose(const Tensor& actual, const Tensor& expected, Tolerance tol,
                     const std::string& label) {
  report(compare(actual, expected, tol), label, tol);
}

void assert_equal(std::span<const std::int32_t> actual, std::span<const std::int32_t> expected,
                  const std::string& label) {
  std::ostringstream os;
  if (!label.empty()) os << label << ": ";

  if (actual.size() != expected.size()) {
    os << "length mismatch: actual has " << actual.size() << ", expected has " << expected.size();
    CHECK_MESSAGE(false, os.str());
    return;
  }

  std::int64_t first = -1;
  std::int64_t n_diff = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (actual[i] != expected[i]) {
      ++n_diff;
      if (first < 0) first = static_cast<std::int64_t>(i);
    }
  }
  if (n_diff > 0) {
    os << n_diff << " of " << expected.size() << " values differ; first at index " << first
       << ": actual " << actual[static_cast<std::size_t>(first)] << " expected "
       << expected[static_cast<std::size_t>(first)];
    CHECK_MESSAGE(false, os.str());
    return;
  }
  if (expected.empty()) {
    os << "compared zero ids — the assertion did nothing";
    CHECK_MESSAGE(false, os.str());
    return;
  }
  CHECK(n_diff == 0);  // register the passing comparison, see report() above
}

}  // namespace minillm::test
