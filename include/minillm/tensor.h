// minillm — tensor.h
//
// A deliberately tiny dense-tensor type. It is a *handle*: shape, strides,
// dtype and a pointer into some storage. Copying a Tensor is cheap and shares
// the storage (views keep the owning buffer alive via a shared reference).
//
// Global conventions (see docs/ARCHITECTURE.md):
//   * Row-major. strides() are in ELEMENTS, not bytes.
//   * Rank is 1..4. Rank 0 means "no tensor" (default-constructed, is_empty()).
//     Scalars are not modelled; use shape {1}.
//   * All compute happens in F32. BF16/F16 exist only so the loader can read
//     checkpoint bytes before converting; Q8_0 is reserved and unimplemented.
//   * Deliberately absent: broadcasting, autograd, arithmetic operator
//     overloading, lazy evaluation, device/stream abstraction, type erasure
//     beyond the dtype tag. Ops are free functions in ops.h.
//
// THIS HEADER IS FROZEN. See docs/ARCHITECTURE.md.

#ifndef MINILLM_TENSOR_H
#define MINILLM_TENSOR_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>

namespace minillm {

/// Element types. F32 is the only type ops operate on.
enum class DType : int {
  F32 = 0,   ///< IEEE-754 binary32, the compute type.
  BF16 = 1,  ///< bfloat16, as stored in HF safetensors checkpoints.
  F16 = 2,   ///< IEEE-754 binary16, accepted by the loader.
  Q8_0 = 3,  ///< Reserved for a future quantized path. Not implemented.
};

/// Maximum tensor rank.
inline constexpr int kMaxDims = 4;

/// Size of one element in bytes. Throws std::invalid_argument for Q8_0.
std::size_t dtype_size(DType dt);

/// Lowercase name of the dtype ("f32", "bf16", "f16", "q8_0").
const char* dtype_name(DType dt);

/// Maps a safetensors dtype string ("F32", "BF16", "F16") to a DType.
/// Throws std::invalid_argument for anything else.
DType dtype_from_string(std::string_view s);

/// bf16 -> f32. Exact: bf16 is the top 16 bits of a f32, so this is a shift.
inline float bf16_to_f32(std::uint16_t v) {
  const std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

/// f32 -> bf16, round-to-nearest-even. Only used by tests that fabricate
/// checkpoint bytes; the inference path never narrows.
inline std::uint16_t f32_to_bf16(float v) {
  std::uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  // NaN: keep it a NaN after truncation.
  if ((bits & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>((bits >> 16) | 0x0040u);
  const std::uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>((bits + rounding_bias) >> 16);
}

/// f16 -> f32. Exact (every f16 value is representable in f32).
inline float f16_to_f32(std::uint16_t v) {
  const std::uint32_t sign = static_cast<std::uint32_t>(v & 0x8000u) << 16;
  const std::uint32_t exp = (v >> 10) & 0x1fu;
  const std::uint32_t mant = v & 0x3ffu;
  std::uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // +-0
    } else {
      // Subnormal f16: renormalize into f32's exponent range.
      std::uint32_t e = exp;
      std::uint32_t m = mant;
      int shift = 0;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        ++shift;
      }
      m &= 0x3ffu;
      e = 127 - 15 - static_cast<std::uint32_t>(shift) + 1;
      bits = sign | (e << 23) | (m << 13);
    }
  } else if (exp == 0x1fu) {
    bits = sign | 0x7f800000u | (mant << 13);  // inf / nan
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

/// A rank and up to kMaxDims extents. Row-major, outermost dim first.
/// operator[] and operator== are structural accessors only; Shape has no
/// arithmetic.
class Shape {
 public:
  Shape() = default;
  Shape(std::initializer_list<std::int64_t> dims);
  Shape(const std::int64_t* dims, int ndim);

  int ndim() const { return ndim_; }
  /// Extent of dimension i. Throws std::out_of_range if i is not in [0, ndim).
  std::int64_t operator[](int i) const;
  std::int64_t& operator[](int i);

  /// Product of the extents; 0 for rank 0 (a rank-0 Shape means "no shape").
  std::int64_t numel() const;

  bool operator==(const Shape& other) const;
  bool operator!=(const Shape& other) const { return !(*this == other); }

  /// "[2, 3, 4]" — for error messages.
  std::string to_string() const;

 private:
  std::array<std::int64_t, kMaxDims> dims_{};
  int ndim_ = 0;
};

/// Dense strided tensor handle.
///
/// Ownership: `zeros`/`empty` allocate a shared buffer; `wrap` points at memory
/// someone else owns. Copies share storage, and any view produced by
/// `reshape`/`slice`/`row` keeps the originating buffer alive. `owns_data()`
/// reports whether this handle holds a reference to such a buffer.
class Tensor {
 public:
  /// An empty handle: rank 0, null data.
  Tensor() = default;

  /// Owning allocation, zero-filled. `dtype` must not be Q8_0.
  static Tensor zeros(const Shape& shape, DType dtype = DType::F32);
  /// Owning allocation, uninitialized.
  static Tensor empty(const Shape& shape, DType dtype = DType::F32);
  /// Non-owning, contiguous view over caller-owned memory. The caller must
  /// keep `data` alive for as long as the Tensor or any of its views is used.
  /// For const memory, const_cast at the call site; Tensor does not track
  /// constness and ops writing to a view built from const memory is UB.
  static Tensor wrap(void* data, const Shape& shape, DType dtype = DType::F32);

  const Shape& shape() const { return shape_; }
  int ndim() const { return shape_.ndim(); }
  /// Extent of dimension i. Throws std::out_of_range if out of range.
  std::int64_t dim(int i) const { return shape_[i]; }
  std::int64_t numel() const { return shape_.numel(); }
  DType dtype() const { return dtype_; }

  /// Strides in elements. Entries at index >= ndim() are unspecified.
  const std::array<std::int64_t, kMaxDims>& strides() const { return strides_; }
  /// Stride of dimension i in elements. Throws std::out_of_range.
  std::int64_t stride(int i) const;

  /// True if the strides are the row-major packed strides for this shape.
  bool is_contiguous() const;
  /// True if this handle references an allocation it keeps alive.
  bool owns_data() const { return storage_ != nullptr; }
  /// True for a default-constructed handle (rank 0 / null data).
  bool is_empty() const { return data_ == nullptr || shape_.ndim() == 0; }

  void* data() { return data_; }
  const void* data() const { return data_; }

  /// Unchecked typed pointer. Prefer f32() in code that assumes F32.
  template <class T>
  T* data_as() {
    return static_cast<T*>(data_);
  }
  template <class T>
  const T* data_as() const {
    return static_cast<const T*>(data_);
  }

  /// Typed pointer, checked: throws std::invalid_argument if dtype() != F32.
  float* f32();
  const float* f32() const;

  /// Element offset (NOT bytes) of the given index, using strides. Indices
  /// beyond ndim() must be 0. Bounds are not checked — this is the inner-loop
  /// addressing helper.
  std::int64_t offset(std::int64_t i0, std::int64_t i1 = 0, std::int64_t i2 = 0,
                      std::int64_t i3 = 0) const {
    return i0 * strides_[0] + i1 * strides_[1] + i2 * strides_[2] + i3 * strides_[3];
  }

  /// View with a different shape, same element count. Requires is_contiguous().
  /// Throws std::invalid_argument otherwise.
  Tensor reshape(const Shape& shape) const;

  /// View of `count` entries along dimension `dim`, starting at `start`. Rank
  /// is preserved. Slicing dimension 0 of a contiguous tensor stays
  /// contiguous; slicing any other dimension generally does not.
  /// Throws std::out_of_range if the range is out of bounds.
  Tensor slice(int dim, std::int64_t start, std::int64_t count) const;

  /// View of index `index` along dimension 0, with that dimension dropped
  /// (rank - 1). Requires ndim() >= 2. Throws std::out_of_range / std::invalid_argument.
  Tensor row(std::int64_t index) const;

  /// Owning, contiguous copy with the same shape and dtype. The only method
  /// that copies element data; use it to materialize a non-contiguous view.
  Tensor copy_contiguous() const;

  /// "f32[2, 3] (contiguous)" — for error messages and test output.
  std::string to_string() const;

 private:
  std::shared_ptr<std::byte[]> storage_;  ///< null for non-owning views
  void* data_ = nullptr;                  ///< first element
  Shape shape_;
  std::array<std::int64_t, kMaxDims> strides_{};
  DType dtype_ = DType::F32;
};

}  // namespace minillm

#endif  // MINILLM_TENSOR_H
