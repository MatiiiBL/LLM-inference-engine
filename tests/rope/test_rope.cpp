// Workflow 05-rope owns this file.
//
// ops::rope against ref::rope (random inputs), against transformers (the
// golden data), and against the properties a rotation must have. The ones
// that matter most: the rotate_half pairing (i with i + head_dim/2) and that
// argument validation happens before anything is written.

#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/allclose.h"
#include "common/oracle.h"
#include "common/random.h"
#include "minillm/ops.h"

using namespace minillm;
using namespace minillm::test;

namespace {

bool bitwise_equal(const Tensor& a, const Tensor& b) {
  return a.shape() == b.shape() &&
         std::memcmp(a.f32(), b.f32(), static_cast<std::size_t>(a.numel()) * sizeof(float)) == 0;
}

// The [cos | sin] row the contract defines for one position, computed here
// independently of ops::rope: the float32 inv_freq chain from ops.h, a float32
// angle, then cos and sin in double rounded to float.
std::vector<float> expected_row(std::int64_t head_dim, float theta, std::int64_t pos) {
  const std::int64_t half = head_dim / 2;
  std::vector<float> row(static_cast<std::size_t>(head_dim));
  for (std::int64_t i = 0; i < half; ++i) {
    const float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
    const float inv_freq = 1.0f / std::pow(theta, exponent);
    const float angle = static_cast<float>(pos) * inv_freq;
    row[static_cast<std::size_t>(i)] = static_cast<float>(std::cos(static_cast<double>(angle)));
    row[static_cast<std::size_t>(i + half)] =
        static_cast<float>(std::sin(static_cast<double>(angle)));
  }
  return row;
}

// [n_tokens, n_heads, head_dim] with every head [1 ... 1 | 0 ... 0]. Rotating
// it writes x[i] = 1*cos - 0*sin = cos and x[i + half] = 0*cos + 1*sin = sin
// exactly, fused multiply-add or not, so the output IS the cos/sin row and can
// be compared bit for bit.
Tensor unit_heads(std::int64_t n_tokens, std::int64_t n_heads, std::int64_t head_dim) {
  Tensor t = Tensor::zeros({n_tokens, n_heads, head_dim});
  for (std::int64_t r = 0; r < n_tokens * n_heads; ++r) {
    for (std::int64_t i = 0; i < head_dim / 2; ++i) t.f32()[r * head_dim + i] = 1.0f;
  }
  return t;
}

// Rotates unit heads at `positions` and counts elements that differ from
// expected_row in any bit.
std::int64_t row_mismatches(std::int64_t head_dim, float theta,
                            const std::vector<std::int64_t>& positions) {
  const auto n_tokens = static_cast<std::int64_t>(positions.size());
  const std::int64_t n_heads = 2;
  Tensor q = unit_heads(n_tokens, n_heads, head_dim);
  ops::rope(q, nullptr, positions, theta);

  std::int64_t mismatches = 0;
  for (std::int64_t t = 0; t < n_tokens; ++t) {
    const std::vector<float> row =
        expected_row(head_dim, theta, positions[static_cast<std::size_t>(t)]);
    for (std::int64_t h = 0; h < n_heads; ++h) {
      const float* head = q.f32() + (t * n_heads + h) * head_dim;
      if (std::memcmp(head, row.data(), row.size() * sizeof(float)) != 0) ++mismatches;
    }
  }
  return mismatches;
}

}  // namespace

TEST_CASE("ops::rope agrees with ref::rope at the model's shapes") {
  // 9 query heads, 3 KV heads, head_dim 64, rope_theta 100000.
  const float theta = 100000.0f;
  for (const std::int64_t n_tokens : {1, 8}) {
    for (const std::int64_t start : {0, 1, 137, 4000}) {
      std::vector<std::int64_t> positions(static_cast<std::size_t>(n_tokens));
      for (std::int64_t t = 0; t < n_tokens; ++t) positions[static_cast<std::size_t>(t)] = start + t;

      Tensor q_actual = random_tensor({n_tokens, 9, 64}, 301);
      Tensor k_actual = random_tensor({n_tokens, 3, 64}, 302);
      Tensor q_expected = q_actual.copy_contiguous();
      Tensor k_expected = k_actual.copy_contiguous();

      ref::rope(q_expected, &k_expected, positions, theta);
      ops::rope(q_actual, &k_actual, positions, theta);

      // ops:: vs ref:: needs no position allowance: both run the same inv_freq
      // chain in the same process, so only reassociation can differ.
      const std::string label = "ops::rope vs ref n=" + std::to_string(n_tokens) + " start=" +
                                std::to_string(start);
      assert_allclose(q_actual, q_expected, tol::kRope, label + " q");
      assert_allclose(k_actual, k_expected, tol::kRope, label + " k");
    }
  }
}

TEST_CASE("ops::rope handles a single token and q alone") {
  const std::vector<std::int64_t> positions{42};
  Tensor q_actual = random_tensor({1, 9, 64}, 303);
  Tensor q_expected = q_actual.copy_contiguous();
  ref::rope(q_expected, nullptr, positions, 100000.0f);
  ops::rope(q_actual, nullptr, positions, 100000.0f);
  assert_allclose(q_actual, q_expected, tol::kRope, "ops::rope q only");
}

TEST_CASE("ops::rope matches transformers") {
  for (const std::string variant : {"small", "model", "offset"}) {
    const std::string base = "ops/rope_" + variant + "_";
    MINILLM_SKIP_WITHOUT_ORACLE({base + "q.npy", base + "k.npy", base + "positions.npy",
                                 base + "theta.npy", base + "q_out.npy", base + "k_out.npy"});

    Tensor q = oracle_npy(base + "q.npy").to_tensor();
    Tensor k = oracle_npy(base + "k.npy").to_tensor();
    const std::vector<std::int64_t> positions = oracle_npy(base + "positions.npy").as_i64();
    const std::vector<float> theta = oracle_npy(base + "theta.npy").as_f32();
    const Tensor q_expected = oracle_npy(base + "q_out.npy").to_tensor();
    const Tensor k_expected = oracle_npy(base + "k_out.npy").to_tensor();
    REQUIRE(theta.size() == 1);

    const std::int64_t max_position = positions.empty() ? 0 : positions.back();
    const Tolerance rope_tol = rope_tolerance(max_position);

    ops::rope(q, &k, positions, theta[0]);
    assert_allclose(q, q_expected, rope_tol, "ops::rope " + variant + " q");
    assert_allclose(k, k_expected, rope_tol, "ops::rope " + variant + " k");
  }
}

TEST_CASE("ops::rope agrees with ref::rope on unordered and repeated positions") {
  // The model always passes start_pos + [0..n), but the contract is any
  // positions >= 0, in any order — including ones far past any context.
  const std::vector<std::int64_t> positions{4000, 0, 137, 137, 1, 20000, 1 << 20};
  Tensor q_actual = random_tensor({7, 9, 64}, 304);
  Tensor k_actual = random_tensor({7, 3, 64}, 305);
  Tensor q_expected = q_actual.copy_contiguous();
  Tensor k_expected = k_actual.copy_contiguous();

  ref::rope(q_expected, &k_expected, positions, 100000.0f);
  ops::rope(q_actual, &k_actual, positions, 100000.0f);
  assert_allclose(q_actual, q_expected, tol::kRope, "ops::rope unordered q");
  assert_allclose(k_actual, k_expected, tol::kRope, "ops::rope unordered k");
}

TEST_CASE("ops::rope pairs dimension i with i + head_dim/2") {
  // One token at position 1, one head, head_dim 4 => half = 2, and
  // inv_freq[0] = 1 so the first pair rotates by exactly 1 radian. With a
  // unit input, x0 * cos - x2 * sin is exactly cos, so this compares exactly.
  Tensor q = Tensor::zeros({1, 1, 4});
  q.f32()[0] = 1.0f;

  const std::vector<std::int64_t> positions{1};
  ops::rope(q, nullptr, positions, 10000.0f);

  // Dimension 0 pairs with dimension 2: x0 -> x0*cos - x2*sin, x2 -> x2*cos + x0*sin.
  CHECK(q.f32()[0] == static_cast<float>(std::cos(1.0)));
  CHECK(q.f32()[2] == static_cast<float>(std::sin(1.0)));
  // The interleaved convention would have put sin(1) in dimension 1.
  CHECK(q.f32()[1] == 0.0f);
  CHECK(q.f32()[3] == 0.0f);
}

TEST_CASE("ops::rope at position 0 is the identity") {
  Tensor q = random_tensor({3, 9, 64}, 306);
  Tensor k = random_tensor({3, 3, 64}, 307);
  const Tensor q_before = q.copy_contiguous();
  const Tensor k_before = k.copy_contiguous();

  const std::vector<std::int64_t> positions{0, 0, 0};
  ops::rope(q, &k, positions, 100000.0f);
  CHECK(bitwise_equal(q, q_before));
  CHECK(bitwise_equal(k, k_before));
}

TEST_CASE("ops::rope preserves the norm of each rotated pair") {
  Tensor q = random_tensor({4, 9, 64}, 308);
  const Tensor before = q.copy_contiguous();
  const std::vector<std::int64_t> positions{0, 1, 137, 4000};
  ops::rope(q, nullptr, positions, 100000.0f);

  const std::int64_t n_heads = 9;
  const std::int64_t head_dim = 64;
  const std::int64_t half = head_dim / 2;
  for (std::int64_t t = 0; t < 4; ++t) {
    for (std::int64_t h = 0; h < n_heads; ++h) {
      for (std::int64_t i = 0; i < half; ++i) {
        const std::int64_t lo = (t * n_heads + h) * head_dim + i;
        const std::int64_t hi = lo + half;
        const double n_before = static_cast<double>(before.f32()[lo]) * before.f32()[lo] +
                                static_cast<double>(before.f32()[hi]) * before.f32()[hi];
        const double n_after = static_cast<double>(q.f32()[lo]) * q.f32()[lo] +
                               static_cast<double>(q.f32()[hi]) * q.f32()[hi];
        CHECK(n_after == doctest::Approx(n_before).epsilon(tol::kRope.rtol));
      }
    }
  }
}

TEST_CASE("ops::rope with k == nullptr rotates q exactly as it would alongside k") {
  const std::vector<std::int64_t> positions{5, 6};
  Tensor q_alone = random_tensor({2, 9, 64}, 309);
  Tensor q_with_k = q_alone.copy_contiguous();
  Tensor k = random_tensor({2, 3, 64}, 310);

  ops::rope(q_alone, nullptr, positions, 100000.0f);
  ops::rope(q_with_k, &k, positions, 100000.0f);
  CHECK(bitwise_equal(q_alone, q_with_k));
}

TEST_CASE("ops::rope validates its arguments") {
  Tensor q = Tensor::zeros({2, 2, 4});
  const std::vector<std::int64_t> two{0, 1};

  // positions must have one entry per token, for q and for k.
  CHECK_THROWS_AS(ops::rope(q, nullptr, std::vector<std::int64_t>{0}, 10000.0f),
                  std::invalid_argument);
  Tensor k_short = Tensor::zeros({1, 1, 4});
  CHECK_THROWS_AS(ops::rope(q, &k_short, two, 10000.0f), std::invalid_argument);
  // head_dim must be even.
  Tensor odd = Tensor::zeros({2, 2, 5});
  CHECK_THROWS_AS(ops::rope(odd, nullptr, two, 10000.0f), std::invalid_argument);
  // rank must be 3.
  Tensor flat = Tensor::zeros({2, 4});
  CHECK_THROWS_AS(ops::rope(flat, nullptr, two, 10000.0f), std::invalid_argument);
  Tensor k_flat = Tensor::zeros({2, 4});
  CHECK_THROWS_AS(ops::rope(q, &k_flat, two, 10000.0f), std::invalid_argument);
  // theta must be positive, and NaN is not positive.
  CHECK_THROWS_AS(ops::rope(q, nullptr, two, 0.0f), std::invalid_argument);
  CHECK_THROWS_AS(ops::rope(q, nullptr, two, -10000.0f), std::invalid_argument);
  CHECK_THROWS_AS(ops::rope(q, nullptr, two, std::nanf("")), std::invalid_argument);
  // negative positions are a bug, not a convention.
  CHECK_THROWS_AS(ops::rope(q, nullptr, std::vector<std::int64_t>{0, -1}, 10000.0f),
                  std::invalid_argument);
  // q and k must agree on head_dim.
  Tensor k_wrong = Tensor::zeros({2, 1, 8});
  CHECK_THROWS_AS(ops::rope(q, &k_wrong, two, 10000.0f), std::invalid_argument);
  // f32 only, and contiguous only.
  Tensor bf16 = Tensor::zeros({2, 2, 4}, DType::BF16);
  CHECK_THROWS_AS(ops::rope(bf16, nullptr, two, 10000.0f), std::invalid_argument);
  Tensor wide = Tensor::zeros({2, 4, 4});
  Tensor strided = wide.slice(1, 0, 2);
  REQUIRE_FALSE(strided.is_contiguous());
  CHECK_THROWS_AS(ops::rope(strided, nullptr, two, 10000.0f), std::invalid_argument);
}

TEST_CASE("ops::rope writes nothing when it throws") {
  // Every argument is checked before the first write, so a bad k or a bad
  // position late in the list cannot leave q half rotated.
  Tensor q = random_tensor({2, 2, 4}, 311);
  Tensor k = random_tensor({2, 1, 4}, 312);
  const Tensor q_before = q.copy_contiguous();
  const Tensor k_before = k.copy_contiguous();

  Tensor k_wrong = Tensor::zeros({2, 1, 8});
  CHECK_THROWS_AS(ops::rope(q, &k_wrong, std::vector<std::int64_t>{3, 4}, 10000.0f),
                  std::invalid_argument);
  CHECK(bitwise_equal(q, q_before));

  CHECK_THROWS_AS(ops::rope(q, &k, std::vector<std::int64_t>{3, -1}, 10000.0f),
                  std::invalid_argument);
  CHECK(bitwise_equal(q, q_before));
  CHECK(bitwise_equal(k, k_before));
}

TEST_CASE("ops::rope agrees with ref::rope on long prefills") {
  // 16384 is where this implementation stops caching rows at head_dim 64, so
  // start 16300 straddles the table and the per-call path in one call.
  for (const std::int64_t start : {0, 16300}) {
    const std::int64_t n_tokens = 200;
    std::vector<std::int64_t> positions(static_cast<std::size_t>(n_tokens));
    for (std::int64_t t = 0; t < n_tokens; ++t) positions[static_cast<std::size_t>(t)] = start + t;

    Tensor q_actual = random_tensor({n_tokens, 9, 64}, 313);
    Tensor k_actual = random_tensor({n_tokens, 3, 64}, 314);
    Tensor q_expected = q_actual.copy_contiguous();
    Tensor k_expected = k_actual.copy_contiguous();

    ref::rope(q_expected, &k_expected, positions, 100000.0f);
    ops::rope(q_actual, &k_actual, positions, 100000.0f);
    const std::string label = "ops::rope prefill start=" + std::to_string(start);
    assert_allclose(q_actual, q_expected, tol::kRope, label + " q");
    assert_allclose(k_actual, k_expected, tol::kRope, label + " k");
  }
}

TEST_CASE("ops::rope's cached cos/sin rows are bit-identical to recomputing them") {
  // Each position is visited cold, warm, and inside a mixed batch, so a row is
  // read from a freshly built table, from a grown copy of one, and from the
  // per-call path past the table's reach; every one must equal the row the
  // contract defines, in every bit. (6, 12345) is a key no other test uses, so
  // its first visit really is cold.
  struct Key {
    std::int64_t head_dim;
    float theta;
  };
  const std::vector<std::int64_t> positions{0,     1,     63,    64,      65,      137,    4000,
                                            8191,  8192,  16383, 16384,   20000,   1 << 20,
                                            1 << 24};
  for (const Key key : {Key{6, 12345.0f}, Key{64, 100000.0f}, Key{128, 500000.0f}}) {
    CAPTURE(key.head_dim);
    CAPTURE(key.theta);
    for (const std::int64_t pos : positions) {
      CAPTURE(pos);
      CHECK(row_mismatches(key.head_dim, key.theta, {pos}) == 0);
    }
    for (auto it = positions.rbegin(); it != positions.rend(); ++it) {
      CAPTURE(*it);
      CHECK(row_mismatches(key.head_dim, key.theta, {*it}) == 0);
    }
    CHECK(row_mismatches(key.head_dim, key.theta, positions) == 0);
  }

  // More keys than the cache keeps, so some are evicted and rebuilt.
  for (int round = 0; round < 2; ++round) {
    for (const float theta : {1000.0f, 2000.0f, 3000.0f, 4000.0f, 5000.0f, 6000.0f}) {
      CAPTURE(theta);
      CHECK(row_mismatches(16, theta, {0, 9, 300}) == 0);
    }
  }
}

TEST_CASE("ops::rope is safe to call concurrently") {
  // Threads sweep interleaved positions on keys that start cold, so tables
  // are built, grown and swapped while other threads are reading them. Under
  // tsan this is the race check; everywhere it checks no thread ever reads a
  // wrong row.
  const int n_threads = 8;
  std::vector<std::int64_t> mismatches(n_threads, 0);
  std::vector<std::thread> threads;
  for (int j = 0; j < n_threads; ++j) {
    threads.emplace_back([j, &mismatches] {
      const float theta = j % 2 == 0 ? 31337.0f : 71717.0f;
      for (std::int64_t pos = j; pos < 4096; pos += 13 * n_threads) {
        mismatches[static_cast<std::size_t>(j)] += row_mismatches(64, theta, {pos, pos + 1});
      }
    });
  }
  for (std::thread& t : threads) t.join();
  for (int j = 0; j < n_threads; ++j) {
    CAPTURE(j);
    CHECK(mismatches[static_cast<std::size_t>(j)] == 0);
  }
}
