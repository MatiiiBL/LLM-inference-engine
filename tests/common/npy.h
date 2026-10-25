// A minimal .npy reader for the golden data.
//
// Supports exactly what oracle/generate.py writes: version 1.0 and 2.0
// headers, little-endian, C order, dtypes float32 ('<f4'), int32 ('<i4') and
// int64 ('<i8'). Anything else throws with a message naming the file — a
// surprising dtype in the oracle is a bug in the generator, not something to
// paper over here.

#ifndef MINILLM_TESTS_COMMON_NPY_H
#define MINILLM_TESTS_COMMON_NPY_H

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "minillm/tensor.h"

namespace minillm::test {

/// A loaded .npy file: its shape, its numpy dtype string, and its raw bytes.
class NpyArray {
 public:
  const std::vector<std::int64_t>& shape() const { return shape_; }
  /// The numpy descr, e.g. "<f4".
  const std::string& descr() const { return descr_; }
  std::int64_t numel() const;

  bool is_f32() const { return descr_ == "<f4"; }
  bool is_i32() const { return descr_ == "<i4"; }
  bool is_i64() const { return descr_ == "<i8"; }

  /// Typed copies. Each throws std::runtime_error if the dtype does not match.
  std::vector<float> as_f32() const;
  std::vector<std::int32_t> as_i32() const;
  /// int64 data, widened from int32 if that is what the file holds — token
  /// ids and positions show up as either depending on how numpy inferred them.
  std::vector<std::int64_t> as_i64() const;
  /// int32 data, narrowed from int64 if needed. Throws if a value does not
  /// fit, which would mean a corrupt oracle rather than a conversion problem.
  std::vector<std::int32_t> as_i32_narrowing() const;

  /// An owning F32 Tensor with this array's shape. Throws if the file is not
  /// float32 or has rank 0 or rank > 4.
  Tensor to_tensor() const;

  friend NpyArray read_npy(const std::filesystem::path& path);

 private:
  std::vector<std::int64_t> shape_;
  std::string descr_;
  std::vector<std::byte> data_;
  std::size_t item_size_ = 0;
};

/// Reads a .npy file. Throws std::runtime_error if it is missing or malformed.
NpyArray read_npy(const std::filesystem::path& path);

}  // namespace minillm::test

#endif  // MINILLM_TESTS_COMMON_NPY_H
