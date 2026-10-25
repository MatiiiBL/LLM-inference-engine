// Tensor is complete as of Phase 0, so these are real tests, not placeholders.

#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "minillm/tensor.h"

using namespace minillm;

TEST_CASE("dtype metadata") {
  CHECK(dtype_size(DType::F32) == 4);
  CHECK(dtype_size(DType::BF16) == 2);
  CHECK(dtype_size(DType::F16) == 2);
  CHECK_THROWS_AS(dtype_size(DType::Q8_0), std::invalid_argument);

  CHECK(std::string(dtype_name(DType::F32)) == "f32");
  CHECK(std::string(dtype_name(DType::BF16)) == "bf16");

  CHECK(dtype_from_string("F32") == DType::F32);
  CHECK(dtype_from_string("BF16") == DType::BF16);
  CHECK(dtype_from_string("F16") == DType::F16);
  CHECK_THROWS_AS(dtype_from_string("I64"), std::invalid_argument);
}

TEST_CASE("bf16 round trip is exact for values representable in bf16") {
  // bf16 keeps the top 16 bits of an f32: 1 sign, 8 exponent, 7 mantissa.
  // Values needing no more than 7 explicit mantissa bits survive a round trip
  // exactly, at any exponent. 1 + 2^-7 is the smallest step above 1.
  for (const float v : {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, -0.25f, 256.0f, 1.0078125f,
                        std::ldexp(-1.5f, 100), std::ldexp(1.0078125f, -60)}) {
    CHECK(bf16_to_f32(f32_to_bf16(v)) == v);
  }
  // Rounding is to nearest even, so a value needing more mantissa lands close.
  const float x = 3.14159265f;
  CHECK(std::abs(bf16_to_f32(f32_to_bf16(x)) - x) < 0.02f);
  CHECK(bf16_to_f32(0x0000) == 0.0f);
}

TEST_CASE("f16 conversion covers normals, subnormals and specials") {
  CHECK(f16_to_f32(0x0000) == 0.0f);
  CHECK(f16_to_f32(0x3C00) == 1.0f);
  CHECK(f16_to_f32(0xBC00) == -1.0f);
  CHECK(f16_to_f32(0x4000) == 2.0f);
  CHECK(f16_to_f32(0x7C00) == std::numeric_limits<float>::infinity());
  CHECK(std::isnan(f16_to_f32(0x7E00)));
  // Smallest positive subnormal: 2^-24.
  CHECK(f16_to_f32(0x0001) == doctest::Approx(5.9604645e-8f));
  // Largest subnormal, just below 2^-14.
  CHECK(f16_to_f32(0x03FF) == doctest::Approx(6.0975552e-5f));
}

TEST_CASE("Shape basics") {
  const Shape s{2, 3, 4};
  CHECK(s.ndim() == 3);
  CHECK(s[0] == 2);
  CHECK(s[2] == 4);
  CHECK(s.numel() == 24);
  CHECK(s.to_string() == "[2, 3, 4]");
  CHECK_THROWS_AS(s[3], std::out_of_range);
  CHECK_THROWS_AS(s[-1], std::out_of_range);

  CHECK(s == Shape{2, 3, 4});
  CHECK(s != Shape{2, 3});
  CHECK(s != Shape{2, 4, 3});

  // Rank 0 means "no shape", not "scalar".
  const Shape empty;
  CHECK(empty.ndim() == 0);
  CHECK(empty.numel() == 0);

  CHECK_THROWS_AS((Shape{1, 2, 3, 4, 5}), std::invalid_argument);

  const std::int64_t dims[2] = {5, 6};
  const Shape from_ptr(dims, 2);
  CHECK(from_ptr == Shape{5, 6});
}

TEST_CASE("default Tensor is empty") {
  const Tensor t;
  CHECK(t.is_empty());
  CHECK(t.ndim() == 0);
  CHECK(t.numel() == 0);
  CHECK(t.data() == nullptr);
  CHECK_FALSE(t.owns_data());
}

TEST_CASE("zeros and empty allocate owning storage") {
  Tensor t = Tensor::zeros({2, 3});
  CHECK(t.owns_data());
  CHECK_FALSE(t.is_empty());
  CHECK(t.numel() == 6);
  CHECK(t.dtype() == DType::F32);
  CHECK(t.is_contiguous());
  CHECK(t.stride(0) == 3);
  CHECK(t.stride(1) == 1);
  CHECK_THROWS_AS(t.stride(2), std::out_of_range);
  for (std::int64_t i = 0; i < t.numel(); ++i) CHECK(t.f32()[i] == 0.0f);

  Tensor u = Tensor::empty({4}, DType::BF16);
  CHECK(u.dtype() == DType::BF16);
  CHECK(u.numel() == 4);
  CHECK_THROWS_AS(u.f32(), std::invalid_argument);  // wrong dtype

  CHECK_THROWS_AS(Tensor::zeros({1, 2, 3}, DType::Q8_0), std::invalid_argument);
  CHECK_THROWS_AS(Tensor::zeros(Shape{}), std::invalid_argument);  // rank 0
}

TEST_CASE("wrap is a non-owning view") {
  std::vector<float> data{1, 2, 3, 4, 5, 6};
  Tensor t = Tensor::wrap(data.data(), {2, 3});
  CHECK_FALSE(t.owns_data());
  CHECK(t.f32() == data.data());
  t.f32()[0] = 42.0f;
  CHECK(data[0] == 42.0f);  // writes land in the caller's buffer
}

TEST_CASE("strides and offset address elements row-major") {
  Tensor t = Tensor::zeros({2, 3, 4});
  CHECK(t.stride(0) == 12);
  CHECK(t.stride(1) == 4);
  CHECK(t.stride(2) == 1);
  CHECK(t.offset(0, 0, 0) == 0);
  CHECK(t.offset(1, 0, 0) == 12);
  CHECK(t.offset(0, 2, 3) == 11);
  CHECK(t.offset(1, 2, 3) == 23);
}

TEST_CASE("reshape shares storage and requires a matching element count") {
  Tensor t = Tensor::zeros({2, 6});
  t.f32()[7] = 1.0f;

  Tensor r = t.reshape({3, 4});
  CHECK(r.shape() == Shape{3, 4});
  CHECK(r.f32() == t.f32());  // same storage
  CHECK(r.f32()[7] == 1.0f);
  CHECK(r.owns_data());  // the view keeps the buffer alive

  CHECK_THROWS_AS(t.reshape({5, 5}), std::invalid_argument);
}

TEST_CASE("slice along dimension 0 stays contiguous") {
  Tensor t = Tensor::zeros({4, 3});
  for (std::int64_t i = 0; i < 12; ++i) t.f32()[i] = static_cast<float>(i);

  Tensor s = t.slice(0, 1, 2);
  CHECK(s.shape() == Shape{2, 3});
  CHECK(s.is_contiguous());
  CHECK(s.f32()[0] == 3.0f);
  CHECK(s.f32()[5] == 8.0f);

  CHECK_THROWS_AS(t.slice(0, 3, 2), std::out_of_range);
  CHECK_THROWS_AS(t.slice(2, 0, 1), std::out_of_range);
  CHECK_THROWS_AS(t.slice(0, -1, 1), std::out_of_range);
}

TEST_CASE("slice along an inner dimension is non-contiguous") {
  Tensor t = Tensor::zeros({4, 3});
  for (std::int64_t i = 0; i < 12; ++i) t.f32()[i] = static_cast<float>(i);

  Tensor s = t.slice(1, 1, 2);
  CHECK(s.shape() == Shape{4, 2});
  CHECK_FALSE(s.is_contiguous());
  CHECK(s.f32()[s.offset(0, 0)] == 1.0f);
  CHECK(s.f32()[s.offset(1, 0)] == 4.0f);
  CHECK(s.f32()[s.offset(3, 1)] == 11.0f);

  // copy_contiguous is the way back to something the ops will accept.
  Tensor c = s.copy_contiguous();
  CHECK(c.is_contiguous());
  CHECK(c.shape() == s.shape());
  CHECK(c.f32()[0] == 1.0f);
  CHECK(c.f32()[1] == 2.0f);
  CHECK(c.f32()[2] == 4.0f);
  CHECK(c.f32()[7] == 11.0f);
}

TEST_CASE("row drops the leading dimension") {
  Tensor t = Tensor::zeros({3, 2, 4});
  for (std::int64_t i = 0; i < t.numel(); ++i) t.f32()[i] = static_cast<float>(i);

  Tensor r = t.row(1);
  CHECK(r.shape() == Shape{2, 4});
  CHECK(r.is_contiguous());
  CHECK(r.f32()[0] == 8.0f);
  CHECK(r.stride(0) == 4);
  CHECK(r.stride(1) == 1);

  // The KV-cache access pattern: row(layer) then slice(0, 0, length).
  Tensor prefix = t.row(2).slice(0, 0, 1);
  CHECK(prefix.shape() == Shape{1, 4});
  CHECK(prefix.f32()[0] == 16.0f);

  CHECK_THROWS_AS(t.row(3), std::out_of_range);
  CHECK_THROWS_AS(t.row(0).row(0).row(0), std::invalid_argument);  // rank too low
}

TEST_CASE("views keep the owning buffer alive") {
  Tensor view;
  {
    Tensor owner = Tensor::zeros({2, 2});
    owner.f32()[3] = 7.0f;
    view = owner.row(1);
  }  // owner is gone, but its storage is not
  CHECK(view.f32()[1] == 7.0f);
  CHECK(view.owns_data());
}

TEST_CASE("to_string describes the tensor") {
  CHECK(Tensor::zeros({2, 3}).to_string() == "f32[2, 3]");
  std::vector<float> data(6);
  CHECK(Tensor::wrap(data.data(), {2, 3}).to_string() == "f32[2, 3] (view)");
  CHECK(Tensor::zeros({4, 3}).slice(1, 0, 2).to_string() == "f32[4, 2] (strided 3, 1)");
}

// Every shape below is chosen so that the check under test is the only thing
// that can reject it: without the check, the unchecked int64 arithmetic wraps
// (or, under UBSan, aborts) and the call quietly succeeds.

namespace {
constexpr std::int64_t kTwoTo(int n) { return std::int64_t{1} << n; }
}  // namespace

TEST_CASE("Shape::numel throws instead of overflowing") {
  CHECK_THROWS_AS((Shape{kTwoTo(32), kTwoTo(32)}.numel()), std::overflow_error);  // wraps to 0
  CHECK_THROWS_AS((Shape{kTwoTo(62), 2}.numel()), std::overflow_error);           // 2^63
  CHECK(Shape{std::numeric_limits<std::int64_t>::max(), 1}.numel() ==
        std::numeric_limits<std::int64_t>::max());
  // A zero extent makes the count 0, wherever it sits: multiplying left to
  // right would overflow before reaching it.
  CHECK(Shape{kTwoTo(62), kTwoTo(62), 0}.numel() == 0);
  CHECK(Shape{0, kTwoTo(62), kTwoTo(62)}.numel() == 0);
}

TEST_CASE("the factories reject a shape whose element count overflows") {
  // 2^64 elements: an unchecked product wraps to 0, and zeros() would hand
  // back a zero-byte buffer claiming 2^64 elements.
  const Shape huge{kTwoTo(32), kTwoTo(32)};
  const doctest::Contains names_shape("[4294967296, 4294967296]");
  CHECK_THROWS_WITH_AS(Tensor::zeros(huge), names_shape, std::invalid_argument);
  CHECK_THROWS_WITH_AS(Tensor::empty(huge, DType::BF16), names_shape, std::invalid_argument);
  float x = 0.0f;
  CHECK_THROWS_WITH_AS(Tensor::wrap(&x, huge), names_shape, std::invalid_argument);
  // The same overflow in the outer dimensions of a rank-4 shape.
  CHECK_THROWS_AS(Tensor::zeros({kTwoTo(40), kTwoTo(40), 1, 1}), std::invalid_argument);
}

TEST_CASE("an empty shape is valid only if its strides fit") {
  // A zero extent empties the tensor, and zeroes every stride outside it.
  // These two are valid; Arena::tensor relies on the first.
  const Tensor a = Tensor::zeros({kTwoTo(62), 0, kTwoTo(62)});
  CHECK(a.numel() == 0);
  CHECK(a.stride(0) == 0);
  CHECK(a.stride(1) == kTwoTo(62));
  CHECK(a.is_contiguous());
  const Tensor b = Tensor::zeros({kTwoTo(62), kTwoTo(62), 0});
  CHECK(b.numel() == 0);
  CHECK(b.stride(0) == 0);
  // Here dimension 0 would need a stride of 2^124.
  const Shape unrepresentable{0, kTwoTo(62), kTwoTo(62)};
  CHECK_THROWS_WITH_AS(Tensor::zeros(unrepresentable), doctest::Contains("strides"),
                       std::invalid_argument);
  CHECK_THROWS_AS(Tensor::wrap(nullptr, unrepresentable), std::invalid_argument);
}

TEST_CASE("the factories reject a shape whose byte count overflows") {
  // 2^62 f32 elements count fine but take 2^64 bytes, which an unchecked
  // size_t product wraps to 0.
  CHECK_THROWS_WITH_AS(Tensor::zeros({kTwoTo(31), kTwoTo(31)}),
                       doctest::Contains("f32[2147483648, 2147483648]"), std::invalid_argument);
  // The limit is the largest byte count int64 (and ptrdiff_t) can hold, so
  // every byte offset into a tensor fits a signed integer too.
  float x = 0.0f;
  CHECK_THROWS_AS(Tensor::wrap(&x, {kTwoTo(61)}), std::invalid_argument);         // 2^63 bytes
  CHECK_THROWS_AS(Tensor::wrap(&x, {kTwoTo(62)}, DType::BF16), std::invalid_argument);
  CHECK_NOTHROW(Tensor::wrap(&x, {kTwoTo(61) - 1}));  // 2^63 - 4 bytes: no allocation, fits
  CHECK_NOTHROW(Tensor::wrap(&x, {kTwoTo(62) - 1}, DType::BF16));
}

TEST_CASE("reshape rejects a shape whose element count only matches after wrapping") {
  // 11 * 1676976733973595602 = 2^64 + 6, which an unchecked int64 product
  // wraps to exactly the 6 elements this tensor has.
  const Tensor t = Tensor::zeros({2, 3});
  CHECK_THROWS_WITH_AS(t.reshape({11, 1676976733973595602}), doctest::Contains("too large"),
                       std::invalid_argument);
  // 2^64 wraps to 0, the element count of an empty tensor.
  CHECK_THROWS_WITH_AS(Tensor::zeros({0}).reshape({kTwoTo(32), kTwoTo(32)}),
                       doctest::Contains("too large"), std::invalid_argument);
}

TEST_CASE("slice rejects a range whose end overflows") {
  const Tensor t = Tensor::zeros({4});
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  // start + count overflows; wrapped, it is negative and passes "<= 4".
  CHECK_THROWS_AS(t.slice(0, 1, kMax), std::out_of_range);
  CHECK_THROWS_AS(t.slice(0, kMax, 1), std::out_of_range);
  CHECK_THROWS_WITH_AS(t.slice(0, 2, kMax), doctest::Contains("from 2"), std::out_of_range);
  CHECK(t.slice(0, 4, 0).numel() == 0);  // an empty range at the very end is fine
}

TEST_CASE("slicing an empty tensor does not move its pointer") {
  // The buffer is zero bytes long, so any step past its start points outside
  // it. Here the step is 2 elements, past the end of nothing.
  const Tensor e = Tensor::zeros({0, 4});
  CHECK(e.slice(1, 2, 2).data() == e.data());
  // And here (2^61 - 1) * 2 elements of 4 bytes wraps a 64-bit address.
  const Tensor big = Tensor::zeros({0, kTwoTo(61), 2});
  CHECK(big.slice(1, kTwoTo(61) - 1, 1).data() == big.data());
}
