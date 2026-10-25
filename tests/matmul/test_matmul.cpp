// Workflow 04-matmul owns this file.
//
// ops::matmul is checked two ways: against ref::matmul on random inputs (cheap,
// any shape, including the tails a blocked kernel special-cases) and against
// the torch golden data (ground truth).

#include <doctest/doctest.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/allclose.h"
#include "common/oracle.h"
#include "common/random.h"
#include "common/thread_invariant.h"
#include "minillm/ops.h"

using namespace minillm;
using namespace minillm::test;

namespace {

std::string shape_label(std::int64_t m, std::int64_t k, std::int64_t n) {
  return "m=" + std::to_string(m) + " k=" + std::to_string(k) + " n=" + std::to_string(n);
}

Tensor nan_filled(const Shape& shape) {
  Tensor t = Tensor::empty(shape);
  std::fill_n(t.f32(), t.numel(), std::numeric_limits<float>::quiet_NaN());
  return t;
}

}  // namespace

TEST_CASE("ops::matmul agrees with ref::matmul at the model's shapes") {
  // [m, k] x [n, k] for every linear layer in the forward pass, at m = 1 (the
  // decode step) and m = 8 (a short prefill). o_proj has q_proj's shape and
  // gate_proj has up_proj's, so each distinct shape appears once. lm_head uses
  // the full 49152-row vocabulary; only the golden file trims it.
  struct Case {
    const char* name;
    std::int64_t k;
    std::int64_t n;
  };
  const Case cases[] = {
      {"q_proj/o_proj", 576, 576},     {"k_proj/v_proj", 576, 192},
      {"gate_proj/up_proj", 576, 1536}, {"down_proj", 1536, 576},
      {"lm_head", 576, 49152},
  };

  for (const std::int64_t m : {1, 8}) {
    for (const Case& c : cases) {
      const Tensor x = random_tensor({m, c.k}, 201);
      const Tensor w = random_tensor({c.n, c.k}, 202);
      Tensor expected = Tensor::zeros({m, c.n});
      Tensor actual = Tensor::zeros({m, c.n});

      ref::matmul(x, w, expected);
      ops::matmul(x, w, actual);
      assert_allclose(actual, expected, tol::kMatmul,
                      std::string("ops::matmul vs ref ") + c.name + " " +
                          shape_label(m, c.k, c.n));
    }
  }
}

TEST_CASE("ops::matmul is bitwise identical at every thread count") {
  // Threads split the output features into whole column panels, so every
  // element keeps its single-thread accumulation order: not close, equal.
  struct Case {
    const char* name;
    std::int64_t m;
    std::int64_t k;
    std::int64_t n;
  };
  const Case cases[] = {
      {"prefill q_proj/o_proj", 128, 576, 576},
      {"prefill k_proj/v_proj", 128, 576, 192},
      {"prefill gate_proj/up_proj", 128, 576, 1536},
      {"prefill down_proj", 128, 1536, 576},
      {"decode q_proj/o_proj", 1, 576, 576},
      {"decode k_proj/v_proj", 1, 576, 192},
      {"decode gate_proj/up_proj", 1, 576, 1536},
      {"decode down_proj", 1, 1536, 576},
      {"lm_head", 1, 576, 49152},
      {"n = 1", 5, 37, 1},
      {"n = 7", 128, 576, 7},
      {"n = 49152, a row tail", 3, 64, 49152},
      {"a leftover column in a threaded split", 7, 33, 1537},
  };
  for (const Case& c : cases) {
    const std::string label =
        std::string("ops::matmul ") + c.name + " " + shape_label(c.m, c.k, c.n);
    const Tensor x = random_tensor({c.m, c.k}, 211);
    const Tensor w = random_tensor({c.n, c.k}, 212);
    const Tensor single = check_thread_invariant(label, [&] {
      Tensor out = nan_filled({c.m, c.n});
      ops::matmul(x, w, out);
      return out;
    });
    Tensor expected = Tensor::zeros({c.m, c.n});
    ref::matmul(x, w, expected);
    assert_allclose(single, expected, tol::kMatmul, label + " vs ref");
  }
}

TEST_CASE("ops::matmul handles awkward sizes") {
  // Sizes that are not multiples of any plausible vector width or tile, which
  // is where a hand-written kernel's tail handling goes wrong.
  const std::int64_t sizes[] = {1, 2, 3, 5, 7, 9, 15, 17, 31, 33};
  for (const std::int64_t k : sizes) {
    for (const std::int64_t n : sizes) {
      const Tensor x = random_tensor({3, k}, 203);
      const Tensor w = random_tensor({n, k}, 204);
      Tensor expected = Tensor::zeros({3, n});
      Tensor actual = Tensor::zeros({3, n});
      ref::matmul(x, w, expected);
      ops::matmul(x, w, actual);
      assert_allclose(actual, expected, tol::kMatmul,
                      "ops::matmul vs ref k=" + std::to_string(k) + " n=" + std::to_string(n));
    }
  }
}

TEST_CASE("ops::matmul handles every row count") {
  // The sweep above fixes m = 3. A kernel that blocks rows has a row tail of
  // its own, so walk m past every plausible row block, at a k with both a
  // vector body and a remainder and an n that leaves a column tail as well.
  struct Inner {
    std::int64_t k;
    std::int64_t n;
  };
  const Inner inners[] = {{37, 19}, {576, 64}};
  std::vector<std::int64_t> rows;
  for (std::int64_t m = 1; m <= 17; ++m) rows.push_back(m);
  for (const std::int64_t m : {31, 33, 64}) rows.push_back(m);

  for (const Inner& in : inners) {
    for (const std::int64_t m : rows) {
      const Tensor x = random_tensor({m, in.k}, 205);
      const Tensor w = random_tensor({in.n, in.k}, 206);
      Tensor expected = Tensor::zeros({m, in.n});
      Tensor actual = Tensor::zeros({m, in.n});
      ref::matmul(x, w, expected);
      ops::matmul(x, w, actual);
      assert_allclose(actual, expected, tol::kMatmul,
                      "ops::matmul vs ref " + shape_label(m, in.k, in.n));
    }
  }
}

TEST_CASE("ops::matmul overwrites every output element") {
  // out is fully overwritten, never accumulated into. Starting from NaN means
  // an element that is skipped, or added to instead of stored, cannot pass.
  // k = 0 is an empty sum, which is 0.
  struct Case {
    std::int64_t m;
    std::int64_t k;
    std::int64_t n;
  };
  const Case cases[] = {{1, 576, 576}, {8, 1536, 576}, {5, 9, 7}, {3, 0, 4}};

  for (const Case& c : cases) {
    const Tensor x = random_tensor({c.m, c.k}, 207);
    const Tensor w = random_tensor({c.n, c.k}, 208);
    Tensor expected = Tensor::zeros({c.m, c.n});
    Tensor actual = Tensor::empty({c.m, c.n});
    std::fill_n(actual.f32(), actual.numel(), std::numeric_limits<float>::quiet_NaN());

    ref::matmul(x, w, expected);
    ops::matmul(x, w, actual);
    assert_allclose(actual, expected, tol::kMatmul,
                    "ops::matmul over NaN " + shape_label(c.m, c.k, c.n));
  }
}

TEST_CASE("ops::matmul rejects invalid arguments") {
  const Tensor x = Tensor::zeros({2, 3});
  Tensor out = Tensor::zeros({2, 4});

  // The cases ref::matmul is held to (tests/ops_ref/test_matmul.cpp).
  // w's second dimension is in_features and must match x's width.
  CHECK_THROWS_AS(ops::matmul(x, Tensor::zeros({4, 5}), out), std::invalid_argument);
  // out must be [m, n].
  CHECK_THROWS_AS(ops::matmul(x, Tensor::zeros({5, 3}), out), std::invalid_argument);
  // rank 3 is not a matrix.
  CHECK_THROWS_AS(ops::matmul(Tensor::zeros({2, 2, 2}), Tensor::zeros({2, 2}), out),
                  std::invalid_argument);

  // out has the wrong number of rows.
  Tensor out_rows = Tensor::zeros({3, 4});
  CHECK_THROWS_AS(ops::matmul(x, Tensor::zeros({4, 3}), out_rows), std::invalid_argument);

  // Every operand must be f32.
  Tensor out_bf16 = Tensor::zeros({2, 4}, DType::BF16);
  CHECK_THROWS_AS(ops::matmul(Tensor::zeros({2, 3}, DType::BF16), Tensor::zeros({4, 3}), out),
                  std::invalid_argument);
  CHECK_THROWS_AS(ops::matmul(x, Tensor::zeros({4, 3}, DType::BF16), out), std::invalid_argument);
  CHECK_THROWS_AS(ops::matmul(x, Tensor::zeros({4, 3}), out_bf16), std::invalid_argument);

  // Every operand must be contiguous: a column slice keeps its parent's row
  // stride, so it is not.
  Tensor out_view = Tensor::zeros({2, 8}).slice(1, 0, 4);
  CHECK_THROWS_AS(ops::matmul(Tensor::zeros({2, 6}).slice(1, 0, 3), Tensor::zeros({4, 3}), out),
                  std::invalid_argument);
  CHECK_THROWS_AS(ops::matmul(x, Tensor::zeros({4, 6}).slice(1, 0, 3), out),
                  std::invalid_argument);
  CHECK_THROWS_AS(ops::matmul(x, Tensor::zeros({4, 3}), out_view), std::invalid_argument);
}

TEST_CASE("ops::matmul matches torch") {
  for (const std::string variant : {"small", "qkv", "kv", "mlp_up", "mlp_down", "lm_head"}) {
    const std::string base = "ops/matmul_" + variant + "_";
    MINILLM_SKIP_WITHOUT_ORACLE({base + "x.npy", base + "w.npy", base + "out.npy"});

    const Tensor x = oracle_npy(base + "x.npy").to_tensor();
    const Tensor w = oracle_npy(base + "w.npy").to_tensor();
    const Tensor expected = oracle_npy(base + "out.npy").to_tensor();

    Tensor actual = Tensor::zeros(expected.shape());
    ops::matmul(x, w, actual);
    assert_allclose(actual, expected, tol::kMatmul, "ops::matmul " + variant);
  }
}
