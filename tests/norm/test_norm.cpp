// Workflow 03-norm-elementwise owns this file.
//
// In Phase 0 ops:: forwards to ref::, so these pass trivially. That is the
// point: the harness is already here, and the day you replace a body in
// src/ops/norm.cpp or src/ops/elementwise.cpp these tests start doing work.
// Add cases, do not rewrite the structure.

#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "common/allclose.h"
#include "common/oracle.h"
#include "common/random.h"
#include "minillm/ops.h"

using namespace minillm;
using namespace minillm::test;

namespace {

// The real model's shapes, so the kernels are exercised on the widths they
// will actually see. Nothing here is hardcoded into the engine — these are
// test inputs, and the oracle cases below carry the real config values.
constexpr std::int64_t kHidden = 576;
constexpr std::int64_t kIntermediate = 1536;
constexpr std::int64_t kVocab = 49152;

}  // namespace

TEST_CASE("ops::rmsnorm agrees with ref::rmsnorm") {
  for (const std::int64_t rows : {1, 8}) {
    const Tensor x = random_tensor({rows, kHidden}, 101);
    const Tensor weight = random_tensor_uniform({kHidden}, 102, 0.5f, 1.5f);
    const float eps = 1e-5f;

    Tensor expected = Tensor::zeros({rows, kHidden});
    Tensor actual = Tensor::zeros({rows, kHidden});
    ref::rmsnorm(x, weight, eps, expected);
    ops::rmsnorm(x, weight, eps, actual);
    assert_allclose(actual, expected, tol::kReduction,
                    "ops::rmsnorm vs ref [" + std::to_string(rows) + ", 576]");
  }
}

TEST_CASE("ops::rmsnorm supports aliasing out onto x") {
  Tensor x = random_tensor({4, kHidden}, 103);
  const Tensor weight = random_tensor_uniform({kHidden}, 104, 0.5f, 1.5f);

  Tensor expected = Tensor::zeros({4, kHidden});
  ref::rmsnorm(x, weight, 1e-5f, expected);
  ops::rmsnorm(x, weight, 1e-5f, x);
  assert_allclose(x, expected, tol::kReduction, "ops::rmsnorm in place");
}

TEST_CASE("ops::softmax agrees with ref::softmax") {
  // [9, 128] is an attention-score block; [1, vocab] is what the sampler sees.
  for (const Shape shape : {Shape{9, 128}, Shape{1, kVocab}}) {
    Tensor actual = random_tensor(shape, 105);
    Tensor expected = actual.copy_contiguous();
    ref::softmax(expected);
    ops::softmax(actual);
    assert_allclose(actual, expected, tol::kReduction, "ops::softmax vs ref " + shape.to_string());
  }
}

TEST_CASE("ops::add and ops::silu_mul agree with ref") {
  const Tensor a = random_tensor({8, kHidden}, 106);
  const Tensor b = random_tensor({8, kHidden}, 107);
  Tensor expected_add = Tensor::zeros({8, kHidden});
  Tensor actual_add = Tensor::zeros({8, kHidden});
  ref::add(a, b, expected_add);
  ops::add(a, b, actual_add);
  assert_allclose(actual_add, expected_add, tol::kElementwise, "ops::add vs ref");

  const Tensor gate = random_tensor({8, kIntermediate}, 108);
  const Tensor up = random_tensor({8, kIntermediate}, 109);
  Tensor expected_mul = Tensor::zeros({8, kIntermediate});
  Tensor actual_mul = Tensor::zeros({8, kIntermediate});
  ref::silu_mul(gate, up, expected_mul);
  ops::silu_mul(gate, up, actual_mul);
  assert_allclose(actual_mul, expected_mul, tol::kElementwise, "ops::silu_mul vs ref");
}

TEST_CASE("ops::embedding and ops::argmax agree with ref") {
  const Tensor table = random_tensor({1024, kHidden}, 110);
  const std::vector<std::int32_t> ids = random_ids(8, 1024, 111);
  Tensor expected = Tensor::zeros({8, kHidden});
  Tensor actual = Tensor::zeros({8, kHidden});
  ref::embedding(table, ids, expected);
  ops::embedding(table, ids, actual);
  assert_allclose(actual, expected, tol::kElementwise, "ops::embedding vs ref");

  const Tensor logits = random_tensor({kVocab}, 112);
  CHECK(ops::argmax(logits) == ref::argmax(logits));
}

TEST_CASE("ops::rmsnorm matches torch") {
  for (const std::string variant : {"small", "model"}) {
    const std::string base = "ops/rmsnorm_" + variant + "_";
    MINILLM_SKIP_WITHOUT_ORACLE(
        {base + "x.npy", base + "weight.npy", base + "eps.npy", base + "out.npy"});

    const Tensor x = oracle_npy(base + "x.npy").to_tensor();
    const Tensor weight = oracle_npy(base + "weight.npy").to_tensor();
    const std::vector<float> eps = oracle_npy(base + "eps.npy").as_f32();
    const Tensor expected = oracle_npy(base + "out.npy").to_tensor();
    REQUIRE(eps.size() == 1);

    Tensor actual = Tensor::zeros(expected.shape());
    ops::rmsnorm(x, weight, eps[0], actual);
    assert_allclose(actual, expected, tol::kReduction, "ops::rmsnorm " + variant);
  }
}

TEST_CASE("ops::softmax matches torch") {
  for (const std::string variant : {"small", "model", "wide"}) {
    const std::string base = "ops/softmax_" + variant + "_";
    MINILLM_SKIP_WITHOUT_ORACLE({base + "x.npy", base + "out.npy"});

    Tensor x = oracle_npy(base + "x.npy").to_tensor();
    const Tensor expected = oracle_npy(base + "out.npy").to_tensor();
    ops::softmax(x);
    assert_allclose(x, expected, tol::kReduction, "ops::softmax " + variant);
  }
}

TEST_CASE("ops::add, ops::silu_mul and ops::embedding match torch") {
  for (const std::string variant : {"small", "model"}) {
    {
      const std::string base = "ops/add_" + variant + "_";
      MINILLM_SKIP_WITHOUT_ORACLE({base + "a.npy", base + "b.npy", base + "out.npy"});
      const Tensor a = oracle_npy(base + "a.npy").to_tensor();
      const Tensor b = oracle_npy(base + "b.npy").to_tensor();
      const Tensor expected = oracle_npy(base + "out.npy").to_tensor();
      Tensor actual = Tensor::zeros(expected.shape());
      ops::add(a, b, actual);
      assert_allclose(actual, expected, tol::kElementwise, "ops::add " + variant);
    }
    {
      const std::string base = "ops/silu_mul_" + variant + "_";
      MINILLM_SKIP_WITHOUT_ORACLE({base + "gate.npy", base + "up.npy", base + "out.npy"});
      const Tensor gate = oracle_npy(base + "gate.npy").to_tensor();
      const Tensor up = oracle_npy(base + "up.npy").to_tensor();
      const Tensor expected = oracle_npy(base + "out.npy").to_tensor();
      Tensor actual = Tensor::zeros(expected.shape());
      ops::silu_mul(gate, up, actual);
      assert_allclose(actual, expected, tol::kElementwise, "ops::silu_mul " + variant);
    }
    {
      const std::string base = "ops/embedding_" + variant + "_";
      MINILLM_SKIP_WITHOUT_ORACLE({base + "table.npy", base + "ids.npy", base + "out.npy"});
      const Tensor table = oracle_npy(base + "table.npy").to_tensor();
      const std::vector<std::int32_t> ids = oracle_npy(base + "ids.npy").as_i32_narrowing();
      const Tensor expected = oracle_npy(base + "out.npy").to_tensor();
      Tensor actual = Tensor::zeros(expected.shape());
      ops::embedding(table, ids, actual);
      assert_allclose(actual, expected, tol::kElementwise, "ops::embedding " + variant);
    }
  }
}
