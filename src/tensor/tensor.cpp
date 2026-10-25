#include "minillm/tensor.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace minillm {
namespace {

/// Row-major packed strides for `shape`, in elements.
std::array<std::int64_t, kMaxDims> packed_strides(const Shape& shape) {
  std::array<std::int64_t, kMaxDims> strides{};
  std::int64_t acc = 1;
  for (int i = shape.ndim() - 1; i >= 0; --i) {
    strides[static_cast<std::size_t>(i)] = acc;
    acc *= shape[i];
  }
  // Dimensions past ndim() are never indexed with a non-zero value, but a
  // zero stride keeps Tensor::offset() harmless if they are.
  for (int i = shape.ndim(); i < kMaxDims; ++i) {
    strides[static_cast<std::size_t>(i)] = 0;
  }
  return strides;
}

void check_shape(const Shape& shape) {
  if (shape.ndim() < 1 || shape.ndim() > kMaxDims) {
    throw std::invalid_argument("Tensor: rank must be 1.." + std::to_string(kMaxDims) + ", got " +
                                std::to_string(shape.ndim()));
  }
  for (int i = 0; i < shape.ndim(); ++i) {
    if (shape[i] < 0) {
      throw std::invalid_argument("Tensor: negative extent in shape " + shape.to_string());
    }
  }
}

}  // namespace

std::size_t dtype_size(DType dt) {
  switch (dt) {
    case DType::F32:
      return 4;
    case DType::BF16:
    case DType::F16:
      return 2;
    case DType::Q8_0:
      break;
  }
  throw std::invalid_argument("dtype_size: Q8_0 is reserved and not implemented");
}

const char* dtype_name(DType dt) {
  switch (dt) {
    case DType::F32:
      return "f32";
    case DType::BF16:
      return "bf16";
    case DType::F16:
      return "f16";
    case DType::Q8_0:
      return "q8_0";
  }
  return "<invalid dtype>";
}

DType dtype_from_string(std::string_view s) {
  if (s == "F32" || s == "f32" || s == "float32") return DType::F32;
  if (s == "BF16" || s == "bf16" || s == "bfloat16") return DType::BF16;
  if (s == "F16" || s == "f16" || s == "float16") return DType::F16;
  throw std::invalid_argument("dtype_from_string: unsupported dtype '" + std::string(s) + "'");
}

// --------------------------------------------------------------------------
// Shape
// --------------------------------------------------------------------------

Shape::Shape(std::initializer_list<std::int64_t> dims) {
  if (dims.size() > static_cast<std::size_t>(kMaxDims)) {
    throw std::invalid_argument("Shape: at most " + std::to_string(kMaxDims) +
                                " dimensions, got " + std::to_string(dims.size()));
  }
  ndim_ = static_cast<int>(dims.size());
  std::copy(dims.begin(), dims.end(), dims_.begin());
}

Shape::Shape(const std::int64_t* dims, int ndim) {
  if (ndim < 0 || ndim > kMaxDims) {
    throw std::invalid_argument("Shape: at most " + std::to_string(kMaxDims) +
                                " dimensions, got " + std::to_string(ndim));
  }
  if (ndim > 0 && dims == nullptr) {
    throw std::invalid_argument("Shape: null dims pointer with ndim > 0");
  }
  ndim_ = ndim;
  for (int i = 0; i < ndim; ++i) dims_[static_cast<std::size_t>(i)] = dims[i];
}

std::int64_t Shape::operator[](int i) const {
  if (i < 0 || i >= ndim_) {
    throw std::out_of_range("Shape: dimension " + std::to_string(i) + " out of range for rank " +
                            std::to_string(ndim_));
  }
  return dims_[static_cast<std::size_t>(i)];
}

std::int64_t& Shape::operator[](int i) {
  if (i < 0 || i >= ndim_) {
    throw std::out_of_range("Shape: dimension " + std::to_string(i) + " out of range for rank " +
                            std::to_string(ndim_));
  }
  return dims_[static_cast<std::size_t>(i)];
}

std::int64_t Shape::numel() const {
  if (ndim_ == 0) return 0;
  std::int64_t n = 1;
  for (int i = 0; i < ndim_; ++i) n *= dims_[static_cast<std::size_t>(i)];
  return n;
}

bool Shape::operator==(const Shape& other) const {
  if (ndim_ != other.ndim_) return false;
  for (int i = 0; i < ndim_; ++i) {
    if (dims_[static_cast<std::size_t>(i)] != other.dims_[static_cast<std::size_t>(i)]) {
      return false;
    }
  }
  return true;
}

std::string Shape::to_string() const {
  std::ostringstream os;
  os << '[';
  for (int i = 0; i < ndim_; ++i) {
    if (i > 0) os << ", ";
    os << dims_[static_cast<std::size_t>(i)];
  }
  os << ']';
  return os.str();
}

// --------------------------------------------------------------------------
// Tensor
// --------------------------------------------------------------------------

Tensor Tensor::empty(const Shape& shape, DType dtype) {
  check_shape(shape);
  const std::size_t bytes = static_cast<std::size_t>(shape.numel()) * dtype_size(dtype);
  Tensor t;
  // make_shared<std::byte[]> gives us a correctly-aligned allocation for any
  // scalar type and a single control block; views share ownership through it.
  t.storage_ = std::make_shared<std::byte[]>(bytes);
  t.data_ = t.storage_.get();
  t.shape_ = shape;
  t.strides_ = packed_strides(shape);
  t.dtype_ = dtype;
  return t;
}

Tensor Tensor::zeros(const Shape& shape, DType dtype) {
  Tensor t = empty(shape, dtype);
  const std::size_t bytes = static_cast<std::size_t>(shape.numel()) * dtype_size(dtype);
  std::memset(t.data_, 0, bytes);
  return t;
}

Tensor Tensor::wrap(void* data, const Shape& shape, DType dtype) {
  check_shape(shape);
  (void)dtype_size(dtype);  // reject Q8_0 here too
  if (data == nullptr && shape.numel() != 0) {
    throw std::invalid_argument("Tensor::wrap: null data for non-empty shape " + shape.to_string());
  }
  Tensor t;
  t.data_ = data;
  t.shape_ = shape;
  t.strides_ = packed_strides(shape);
  t.dtype_ = dtype;
  return t;
}

std::int64_t Tensor::stride(int i) const {
  if (i < 0 || i >= shape_.ndim()) {
    throw std::out_of_range("Tensor::stride: dimension " + std::to_string(i) +
                            " out of range for " + to_string());
  }
  return strides_[static_cast<std::size_t>(i)];
}

bool Tensor::is_contiguous() const {
  const std::array<std::int64_t, kMaxDims> packed = packed_strides(shape_);
  for (int i = 0; i < shape_.ndim(); ++i) {
    // A dimension of extent 1 can carry any stride without changing the
    // addresses that are actually reachable.
    if (shape_[i] == 1) continue;
    if (strides_[static_cast<std::size_t>(i)] != packed[static_cast<std::size_t>(i)]) return false;
  }
  return true;
}

float* Tensor::f32() {
  if (dtype_ != DType::F32) {
    throw std::invalid_argument("Tensor::f32: tensor is " + std::string(dtype_name(dtype_)) +
                                ", not f32");
  }
  return static_cast<float*>(data_);
}

const float* Tensor::f32() const {
  if (dtype_ != DType::F32) {
    throw std::invalid_argument("Tensor::f32: tensor is " + std::string(dtype_name(dtype_)) +
                                ", not f32");
  }
  return static_cast<const float*>(data_);
}

Tensor Tensor::reshape(const Shape& shape) const {
  check_shape(shape);
  if (!is_contiguous()) {
    throw std::invalid_argument("Tensor::reshape: tensor is not contiguous: " + to_string());
  }
  if (shape.numel() != shape_.numel()) {
    throw std::invalid_argument("Tensor::reshape: " + shape_.to_string() + " -> " +
                                shape.to_string() + " changes the element count");
  }
  Tensor t = *this;
  t.shape_ = shape;
  t.strides_ = packed_strides(shape);
  return t;
}

Tensor Tensor::slice(int dim, std::int64_t start, std::int64_t count) const {
  if (dim < 0 || dim >= shape_.ndim()) {
    throw std::out_of_range("Tensor::slice: dimension " + std::to_string(dim) +
                            " out of range for " + to_string());
  }
  if (start < 0 || count < 0 || start + count > shape_[dim]) {
    throw std::out_of_range("Tensor::slice: [" + std::to_string(start) + ", " +
                            std::to_string(start + count) + ") out of range for dimension " +
                            std::to_string(dim) + " of " + to_string());
  }
  Tensor t = *this;
  t.shape_[dim] = count;
  t.data_ = static_cast<std::byte*>(data_) +
            static_cast<std::size_t>(start * strides_[static_cast<std::size_t>(dim)]) *
                dtype_size(dtype_);
  return t;
}

Tensor Tensor::row(std::int64_t index) const {
  if (shape_.ndim() < 2) {
    throw std::invalid_argument("Tensor::row: needs rank >= 2, got " + to_string());
  }
  if (index < 0 || index >= shape_[0]) {
    throw std::out_of_range("Tensor::row: index " + std::to_string(index) +
                            " out of range for " + to_string());
  }
  std::int64_t dims[kMaxDims];
  for (int i = 1; i < shape_.ndim(); ++i) dims[i - 1] = shape_[i];

  Tensor t;
  t.storage_ = storage_;
  t.data_ = static_cast<std::byte*>(data_) +
            static_cast<std::size_t>(index * strides_[0]) * dtype_size(dtype_);
  t.shape_ = Shape(dims, shape_.ndim() - 1);
  for (int i = 1; i < shape_.ndim(); ++i) {
    t.strides_[static_cast<std::size_t>(i - 1)] = strides_[static_cast<std::size_t>(i)];
  }
  for (int i = shape_.ndim() - 1; i < kMaxDims; ++i) {
    t.strides_[static_cast<std::size_t>(i)] = 0;
  }
  t.dtype_ = dtype_;
  return t;
}

Tensor Tensor::copy_contiguous() const {
  if (is_empty()) return Tensor();
  Tensor out = empty(shape_, dtype_);
  const std::size_t esz = dtype_size(dtype_);
  const auto* src = static_cast<const std::byte*>(data_);
  auto* dst = static_cast<std::byte*>(out.data_);

  // Walk the logical index space; rank is at most 4, so four nested loops
  // cover every case. Missing dimensions have extent 1.
  const std::int64_t d0 = shape_.ndim() > 0 ? shape_[0] : 1;
  const std::int64_t d1 = shape_.ndim() > 1 ? shape_[1] : 1;
  const std::int64_t d2 = shape_.ndim() > 2 ? shape_[2] : 1;
  const std::int64_t d3 = shape_.ndim() > 3 ? shape_[3] : 1;

  std::size_t out_index = 0;
  for (std::int64_t i0 = 0; i0 < d0; ++i0) {
    for (std::int64_t i1 = 0; i1 < d1; ++i1) {
      for (std::int64_t i2 = 0; i2 < d2; ++i2) {
        for (std::int64_t i3 = 0; i3 < d3; ++i3) {
          const std::size_t src_off = static_cast<std::size_t>(offset(i0, i1, i2, i3)) * esz;
          std::memcpy(dst + out_index * esz, src + src_off, esz);
          ++out_index;
        }
      }
    }
  }
  return out;
}

std::string Tensor::to_string() const {
  std::ostringstream os;
  os << dtype_name(dtype_) << shape_.to_string();
  if (!is_contiguous()) {
    os << " (strided";
    for (int i = 0; i < shape_.ndim(); ++i) {
      os << (i == 0 ? " " : ", ") << strides_[static_cast<std::size_t>(i)];
    }
    os << ')';
  }
  if (!owns_data()) os << " (view)";
  return os.str();
}

}  // namespace minillm
