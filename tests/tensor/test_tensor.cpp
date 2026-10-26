// Tensor is complete as of Phase 0, so these are real tests, not placeholders.

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

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

// --------------------------------------------------------------------------
// Q8_0
// --------------------------------------------------------------------------

TEST_CASE("dtype_nbytes counts q8_0 in 34-byte blocks of 32") {
  CHECK(dtype_nbytes(DType::F32, 10) == 40);
  CHECK(dtype_nbytes(DType::BF16, 10) == 20);
  CHECK(dtype_nbytes(DType::Q8_0, 0) == 0);
  CHECK(dtype_nbytes(DType::Q8_0, 32) == 34);
  CHECK(dtype_nbytes(DType::Q8_0, 576 * 1536) == 576 * 48 * 34);
  CHECK_THROWS_AS(dtype_nbytes(DType::Q8_0, 33), std::invalid_argument);
  CHECK_THROWS_AS(dtype_nbytes(DType::F32, -1), std::invalid_argument);
  CHECK_THROWS_AS(dtype_nbytes(DType::F32, kTwoTo(61)), std::invalid_argument);  // 2^63 bytes
  // 2^62 elements are 2^57 blocks of 34 bytes: about 2^62.1, still under 2^63.
  CHECK(dtype_nbytes(DType::Q8_0, kTwoTo(62)) == static_cast<std::size_t>(kTwoTo(57)) * 34u);
}

TEST_CASE("q8_0 tensors need a last dimension that is a multiple of 32") {
  Tensor t = Tensor::zeros({3, 64}, DType::Q8_0);
  CHECK(t.dtype() == DType::Q8_0);
  CHECK(t.numel() == 192);
  CHECK(t.nbytes() == 6 * sizeof(BlockQ8_0));
  CHECK(t.stride(0) == 64);
  CHECK(t.is_contiguous());
  const auto* bytes = t.data_as<unsigned char>();
  for (std::size_t i = 0; i < t.nbytes(); ++i) CHECK(bytes[i] == 0);
  CHECK_THROWS_AS(t.f32(), std::invalid_argument);

  CHECK_NOTHROW(Tensor::empty({32}, DType::Q8_0));
  CHECK_NOTHROW(Tensor::zeros({0, 32}, DType::Q8_0));
  CHECK(Tensor::zeros({4, 0}, DType::Q8_0).nbytes() == 0);  // 0 is a multiple of 32
  CHECK_THROWS_WITH_AS(Tensor::zeros({2, 48}, DType::Q8_0), doctest::Contains("multiple of 32"),
                       std::invalid_argument);
  BlockQ8_0 block{};
  CHECK_THROWS_AS(Tensor::wrap(&block, {32, 1}, DType::Q8_0), std::invalid_argument);
  CHECK(Tensor::wrap(&block, {1, 32}, DType::Q8_0).nbytes() == 34);

  CHECK_NOTHROW(t.reshape({6, 32}));
  CHECK_NOTHROW(t.reshape({192}));
  CHECK_THROWS_AS(t.reshape({12, 16}), std::invalid_argument);
  CHECK_THROWS_AS(t.reshape({64, 3}), std::invalid_argument);
}

TEST_CASE("q8_0 rows and dimension-0 slices start on a block") {
  std::vector<BlockQ8_0> blocks(4 * 3);  // [4, 96]: 3 blocks per row
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    blocks[b].d = static_cast<std::uint16_t>(b);
    for (int j = 0; j < 32; ++j) blocks[b].qs[j] = static_cast<std::int8_t>(j - 16);
  }
  Tensor t = Tensor::wrap(blocks.data(), {4, 96}, DType::Q8_0);

  Tensor r = t.row(2);
  CHECK(r.shape() == Shape{96});
  CHECK(r.data_as<BlockQ8_0>() == blocks.data() + 6);
  CHECK(r.nbytes() == 3 * sizeof(BlockQ8_0));

  Tensor s = t.slice(0, 1, 2);
  CHECK(s.shape() == Shape{2, 96});
  CHECK(s.is_contiguous());
  CHECK(s.data_as<BlockQ8_0>() == blocks.data() + 3);

  Tensor c = s.copy_contiguous();
  CHECK(c.owns_data());
  CHECK(c.dtype() == DType::Q8_0);
  CHECK(c.shape() == Shape{2, 96});
  CHECK(std::memcmp(c.data(), blocks.data() + 3, 6 * sizeof(BlockQ8_0)) == 0);

  // A cut along the last dimension could split a block.
  CHECK_THROWS_AS(t.slice(1, 0, 32), std::invalid_argument);
  CHECK_THROWS_AS(r.slice(0, 32, 32), std::invalid_argument);
  Tensor t3 = Tensor::wrap(blocks.data(), {2, 2, 96}, DType::Q8_0);
  CHECK(t3.slice(0, 1, 1).data_as<BlockQ8_0>() == blocks.data() + 6);
  CHECK_THROWS_AS(t3.slice(1, 1, 1), std::invalid_argument);
}

TEST_CASE("f32_to_f16 rounds to nearest even and inverts f16_to_f32") {
  CHECK(f32_to_f16(1.0f) == 0x3c00);
  CHECK(f32_to_f16(-2.0f) == 0xc000);
  CHECK(f32_to_f16(0.0f) == 0x0000);
  CHECK(f32_to_f16(-0.0f) == 0x8000);
  CHECK(f32_to_f16(65504.0f) == 0x7bff);  // largest finite f16
  CHECK(f32_to_f16(65519.0f) == 0x7bff);  // below the halfway point to 2^16
  CHECK(f32_to_f16(65520.0f) == 0x7c00);  // the halfway point rounds to even: inf
  CHECK(f32_to_f16(std::numeric_limits<float>::infinity()) == 0x7c00);
  CHECK(f32_to_f16(-std::numeric_limits<float>::infinity()) == 0xfc00);
  CHECK(f32_to_f16(std::numeric_limits<float>::quiet_NaN()) == 0x7e00);
  CHECK(f32_to_f16(-std::numeric_limits<float>::quiet_NaN()) == 0xfe00);

  // Ties go to the even mantissa: 1 + 2^-11 is halfway between 1 and 1 + 2^-10.
  CHECK(f32_to_f16(1.0f + std::ldexp(1.0f, -11)) == 0x3c00);
  CHECK(f32_to_f16(1.0f + 3 * std::ldexp(1.0f, -11)) == 0x3c02);
  CHECK(f32_to_f16(std::nextafter(1.0f + std::ldexp(1.0f, -11), 2.0f)) == 0x3c01);

  // Subnormals step by 2^-24, with the same tie rule.
  CHECK(f32_to_f16(std::ldexp(1.0f, -24)) == 0x0001);
  CHECK(f32_to_f16(std::ldexp(1.0f, -25)) == 0x0000);
  CHECK(f32_to_f16(3 * std::ldexp(1.0f, -25)) == 0x0002);
  CHECK(f32_to_f16(std::ldexp(1.0f, -14)) == 0x0400);  // smallest normal

  // Every non-NaN f16 survives f16 -> f32 -> f16.
  int mismatches = 0;
  for (std::uint32_t h = 0; h <= 0xffffu; ++h) {
    const auto bits = static_cast<std::uint16_t>(h);
    if ((bits & 0x7c00u) == 0x7c00u && (bits & 0x03ffu) != 0) continue;  // NaN
    if (f32_to_f16(f16_to_f32(bits)) != bits) ++mismatches;
  }
  CHECK(mismatches == 0);
}

// The hardware f32 -> f16 conversion, rounding to nearest even: fcvt on
// aarch64 (macos-15 in CI), vcvtps2ph on x86-64 with F16C (ubuntu-24.04).
// Hosts with neither skip the comparison below.
namespace {

#if defined(__x86_64__)
#define MINILLM_F16_TARGET __attribute__((target("f16c")))
#else
#define MINILLM_F16_TARGET
#endif

#if defined(__aarch64__)
constexpr const char* kHardwareF16 = "fcvt";
bool hardware_f16_available() { return true; }
std::uint16_t hardware_f32_to_f16(float v) {
  const auto h = static_cast<_Float16>(v);
  std::uint16_t bits;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}
#elif defined(__x86_64__)
constexpr const char* kHardwareF16 = "F16C vcvtps2ph";
bool hardware_f16_available() { return __builtin_cpu_supports("f16c"); }
MINILLM_F16_TARGET std::uint16_t hardware_f32_to_f16(float v) {
  return static_cast<std::uint16_t>(_cvtss_sh(v, _MM_FROUND_TO_NEAREST_INT));
}
#else
constexpr const char* kHardwareF16 = nullptr;
bool hardware_f16_available() { return false; }
std::uint16_t hardware_f32_to_f16(float) { return 0; }
#endif

bool is_f16_nan(std::uint16_t h) { return (h & 0x7c00u) == 0x7c00u && (h & 0x03ffu) != 0; }

/// How many of the bit patterns [first, last], stepping by `step`, f32_to_f16
/// converts differently from the hardware. Compiled for F16C on x86-64 so the
/// hardware conversion inlines into the loop.
MINILLM_F16_TARGET std::uint64_t count_f32_to_f16_mismatches(std::uint32_t first,
                                                             std::uint32_t last,
                                                             std::uint32_t step) {
  std::uint64_t mismatches = 0;
  for (std::uint64_t u = first; u <= last; u += step) {
    const auto bits = static_cast<std::uint32_t>(u);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    const std::uint16_t ours = f32_to_f16(f);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
      // The hardware keeps the top of a NaN's payload; f32_to_f16 promises
      // only "NaN stays NaN", as 0x7e00 with the input's sign.
      const auto nan = static_cast<std::uint16_t>(((bits >> 16) & 0x8000u) | 0x7e00u);
      if (ours != nan || !is_f16_nan(hardware_f32_to_f16(f))) ++mismatches;
    } else if (ours != hardware_f32_to_f16(f)) {
      ++mismatches;
    }
  }
  return mismatches;
}

}  // namespace

TEST_CASE("f32_to_f16 matches the hardware conversion on every non-negative float") {
  if (!hardware_f16_available()) {
    // Both CI runners have one, so a green CI run means both compared.
    if (std::getenv("GITHUB_ACTIONS") != nullptr) {
      FAIL("a CI runner without hardware f32 -> f16 conversion: the comparison did not run");
    }
    MESSAGE("this host has no hardware f32 -> f16 conversion (neither aarch64 nor x86-64 with "
            "F16C); the comparison runs on both CI runners instead");
    return;
  }
  CAPTURE(kHardwareF16);
  // All 2^31 non-negative patterns: zero, subnormals, normals, the overflow to
  // inf, inf itself and every NaN.
  CHECK(count_f32_to_f16_mismatches(0, 0x7fffffffu, 1) == 0);
  // The sign is OR-ed in after rounding the magnitude; a sample of the
  // negative half (every 997th pattern) confirms it.
  CHECK(count_f32_to_f16_mismatches(0x80000000u, 0xffffffffu, 997) == 0);
}
