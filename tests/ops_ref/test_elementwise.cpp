// ref::embedding, ref::silu_mul, ref::add, ref::argmax vs PyTorch.

#include <doctest/doctest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/allclose.h"
#include "common/oracle.h"
#include "common/random.h"
#include "minillm/ops.h"

using namespace minillm;
using namespace minillm::test;

namespace {

/// Runs one oracle case for an op of the form out = f(a, b).
void check_binary(const std::string& op, const std::string& variant,
                  void (*fn)(const Tensor&, const Tensor&, Tensor&), const char* a_name,
                  const char* b_name, Tolerance tol) {
  const std::string base = "ops/" + op + "_" + variant + "_";
  const std::string a_file = base + a_name + ".npy";
  const std::string b_file = base + b_name + ".npy";
  const std::string out_file = base + "out.npy";
  MINILLM_SKIP_WITHOUT_ORACLE({a_file, b_file, out_file});

  const Tensor a = oracle_npy(a_file).to_tensor();
  const Tensor b = oracle_npy(b_file).to_tensor();
  const Tensor expected = oracle_npy(out_file).to_tensor();

  Tensor actual = Tensor::zeros(expected.shape());
  fn(a, b, actual);
  assert_allclose(actual, expected, tol, op + " " + variant + " " + a.shape().to_string());
}

}  // namespace

TEST_CASE("ref::add matches torch") {
  check_binary("add", "small", &ref::add, "a", "b", tol::kElementwise);
  check_binary("add", "model", &ref::add, "a", "b", tol::kElementwise);
}

TEST_CASE("ref::silu_mul matches torch") {
  check_binary("silu_mul", "small", &ref::silu_mul, "gate", "up", tol::kElementwise);
  check_binary("silu_mul", "model", &ref::silu_mul, "gate", "up", tol::kElementwise);
}

TEST_CASE("ref::embedding matches torch") {
  for (const std::string variant : {"small", "model"}) {
    const std::string base = "ops/embedding_" + variant + "_";
    MINILLM_SKIP_WITHOUT_ORACLE({base + "table.npy", base + "ids.npy", base + "out.npy"});

    const Tensor table = oracle_npy(base + "table.npy").to_tensor();
    const std::vector<std::int32_t> ids = oracle_npy(base + "ids.npy").as_i32_narrowing();
    const Tensor expected = oracle_npy(base + "out.npy").to_tensor();

    Tensor actual = Tensor::zeros(expected.shape());
    ref::embedding(table, ids, actual);
    assert_allclose(actual, expected, tol::kElementwise, "embedding " + variant);
  }
}

TEST_CASE("ref::argmax matches torch") {
  for (const std::string variant : {"small", "model"}) {
    const std::string base = "ops/argmax_" + variant + "_";
    MINILLM_SKIP_WITHOUT_ORACLE({base + "x.npy", base + "out.npy"});

    const Tensor x = oracle_npy(base + "x.npy").to_tensor();
    const std::vector<std::int64_t> expected = oracle_npy(base + "out.npy").as_i64();
    REQUIRE(expected.size() == 1);
    CHECK(ref::argmax(x) == static_cast<std::int32_t>(expected[0]));
  }
}

// --- properties that hold without any golden data -------------------------

TEST_CASE("ref::add is exact and supports aliasing") {
  Tensor a = random_tensor({4, 8}, 1);
  const Tensor b = random_tensor({4, 8}, 2);
  const Tensor a_copy = a.copy_contiguous();

  Tensor out = Tensor::zeros({4, 8});
  ref::add(a, b, out);

  // Aliasing out onto a must give the same answer.
  ref::add(a, b, a);
  for (std::int64_t i = 0; i < out.numel(); ++i) {
    CHECK(a.f32()[i] == out.f32()[i]);
    CHECK(out.f32()[i] == a_copy.f32()[i] + b.f32()[i]);  // exact: one fp add
  }
}

TEST_CASE("ref::silu_mul matches the scalar formula and supports aliasing") {
  Tensor gate = random_tensor({3, 5}, 3);
  const Tensor up = random_tensor({3, 5}, 4);
  const Tensor gate_copy = gate.copy_contiguous();

  Tensor out = Tensor::zeros({3, 5});
  ref::silu_mul(gate, up, out);
  for (std::int64_t i = 0; i < out.numel(); ++i) {
    const float v = gate.f32()[i];
    const float want = (v / (1.0f + std::exp(-v))) * up.f32()[i];
    CHECK(out.f32()[i] == want);
  }

  ref::silu_mul(gate, up, gate);
  for (std::int64_t i = 0; i < out.numel(); ++i) CHECK(gate.f32()[i] == out.f32()[i]);
  CHECK(gate_copy.f32()[0] != gate.f32()[0]);  // the copy really was a copy
}

TEST_CASE("ref::embedding gathers rows and rejects bad ids") {
  Tensor table = Tensor::zeros({4, 3});
  for (std::int64_t i = 0; i < table.numel(); ++i) table.f32()[i] = static_cast<float>(i);

  const std::vector<std::int32_t> ids{3, 0, 3};
  Tensor out = Tensor::zeros({3, 3});
  ref::embedding(table, ids, out);
  CHECK(out.f32()[0] == 9.0f);
  CHECK(out.f32()[2] == 11.0f);
  CHECK(out.f32()[3] == 0.0f);
  CHECK(out.f32()[6] == 9.0f);

  // One row out, so the shape check passes and the id check is what fires.
  Tensor one = Tensor::zeros({1, 3});
  CHECK_THROWS_AS(ref::embedding(table, std::vector<std::int32_t>{4}, one), std::out_of_range);
  CHECK_THROWS_AS(ref::embedding(table, std::vector<std::int32_t>{-1}, one), std::out_of_range);

  Tensor wrong = Tensor::zeros({3, 4});
  CHECK_THROWS_AS(ref::embedding(table, ids, wrong), std::invalid_argument);
}

TEST_CASE("ref::argmax resolves ties to the lowest index") {
  Tensor x = Tensor::zeros({5});
  x.f32()[1] = 2.0f;
  x.f32()[3] = 2.0f;
  CHECK(ref::argmax(x) == 1);

  Tensor negative = Tensor::zeros({3});
  negative.f32()[0] = -5.0f;
  negative.f32()[1] = -1.0f;
  negative.f32()[2] = -9.0f;
  CHECK(ref::argmax(negative) == 1);

  CHECK_THROWS_AS(ref::argmax(Tensor::zeros({2, 2})), std::invalid_argument);
}

TEST_CASE("elementwise ops reject mismatched shapes and dtypes") {
  const Tensor a = Tensor::zeros({2, 3});
  const Tensor b = Tensor::zeros({3, 2});
  Tensor out = Tensor::zeros({2, 3});
  CHECK_THROWS_AS(ref::add(a, b, out), std::invalid_argument);

  Tensor bf16 = Tensor::zeros({2, 3}, DType::BF16);
  CHECK_THROWS_AS(ref::add(a, bf16, out), std::invalid_argument);

  // A non-contiguous input is rejected rather than silently mis-read.
  const Tensor strided = Tensor::zeros({2, 6}).slice(1, 0, 3);
  CHECK_THROWS_AS(ref::add(strided, a, out), std::invalid_argument);
}
