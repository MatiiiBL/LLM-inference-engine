// Argument validation for the kernels in norm.cpp and elementwise.cpp.
//
// Private to src/ops/ — not part of the public interface. The same checks as
// src/ops_ref/check.h, which ops.h requires optimized kernels to perform too,
// but with messages that name the ops:: function rather than the reference.

#ifndef MINILLM_SRC_OPS_VALIDATE_H
#define MINILLM_SRC_OPS_VALIDATE_H

#include <stdexcept>
#include <string>

#include "minillm/tensor.h"

namespace minillm::ops::detail {

[[noreturn]] inline void fail(const char* op, const std::string& message) {
  throw std::invalid_argument(std::string("ops::") + op + ": " + message);
}

/// f32 + contiguous, which every op in these files wants of every tensor.
inline void require_f32_contiguous(const char* op, const char* name, const Tensor& t) {
  if (t.dtype() != DType::F32) {
    fail(op, std::string(name) + " must be f32, got " + t.to_string());
  }
  if (!t.is_contiguous()) {
    fail(op, std::string(name) + " must be contiguous, got " + t.to_string());
  }
}

inline void require_f32_contiguous_rank(const char* op, const char* name, const Tensor& t,
                                        int rank) {
  require_f32_contiguous(op, name, t);
  if (t.ndim() != rank) {
    fail(op, std::string(name) + " must have rank " + std::to_string(rank) + ", got " +
                 t.to_string());
  }
}

inline void require_same_shape(const char* op, const char* a_name, const Tensor& a,
                               const char* b_name, const Tensor& b) {
  if (a.shape() != b.shape()) {
    fail(op, std::string(a_name) + " is " + a.shape().to_string() + " but " + b_name + " is " +
                 b.shape().to_string());
  }
}

inline void require_dim(const char* op, const char* name, const Tensor& t, int dim,
                        std::int64_t expected, const char* what) {
  if (t.shape()[dim] != expected) {
    fail(op, std::string(name) + " dimension " + std::to_string(dim) + " is " +
                 std::to_string(t.shape()[dim]) + ", expected " + std::to_string(expected) + " (" +
                 what + ")");
  }
}

}  // namespace minillm::ops::detail

#endif  // MINILLM_SRC_OPS_VALIDATE_H
