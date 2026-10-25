// Shared argument validation for the reference ops.
//
// Private to src/ops_ref/ — not part of the public interface. Every reference
// op validates its arguments up front so a shape bug surfaces as a readable
// exception instead of a wrong number or a crash. The optimized kernels in
// src/ops/ are expected to perform the same checks (they may use their own
// helper; this header is not a contract).

#ifndef MINILLM_SRC_OPS_REF_CHECK_H
#define MINILLM_SRC_OPS_REF_CHECK_H

#include <stdexcept>
#include <string>

#include "minillm/tensor.h"

namespace minillm::ref::detail {

[[noreturn]] inline void fail(const char* op, const std::string& message) {
  throw std::invalid_argument(std::string("ref::") + op + ": " + message);
}

inline void require_f32(const char* op, const char* name, const Tensor& t) {
  if (t.dtype() != DType::F32) {
    fail(op, std::string(name) + " must be f32, got " + t.to_string());
  }
}

inline void require_contiguous(const char* op, const char* name, const Tensor& t) {
  if (!t.is_contiguous()) {
    fail(op, std::string(name) + " must be contiguous, got " + t.to_string());
  }
}

inline void require_rank(const char* op, const char* name, const Tensor& t, int rank) {
  if (t.ndim() != rank) {
    fail(op, std::string(name) + " must have rank " + std::to_string(rank) + ", got " +
                 t.to_string());
  }
}

/// f32 + contiguous + exact rank, the combination almost every op wants.
inline void require_f32_contiguous_rank(const char* op, const char* name, const Tensor& t,
                                        int rank) {
  require_f32(op, name, t);
  require_contiguous(op, name, t);
  require_rank(op, name, t, rank);
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

}  // namespace minillm::ref::detail

#endif  // MINILLM_SRC_OPS_REF_CHECK_H
