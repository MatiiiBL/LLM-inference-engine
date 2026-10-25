// ref::matmul vs PyTorch, at a toy shape and at every shape the real model
// actually uses.

#include <doctest/doctest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "common/allclose.h"
#include "common/oracle.h"
#include "common/random.h"
#include "minillm/ops.h"

using namespace minillm;
using namespace minillm::test;

TEST_CASE("ref::matmul matches torch at the model's real shapes") {
  // One variant per distinct [m, k] x [n, k] in the forward pass. "lm_head"
  // uses a trimmed vocabulary so the golden file stays small; the op does not
  // care how large n is beyond its tail behaviour, which 4096 preserves.
  for (const std::string variant : {"small", "qkv", "kv", "mlp_up", "mlp_down", "lm_head"}) {
    const std::string base = "ops/matmul_" + variant + "_";
    MINILLM_SKIP_WITHOUT_ORACLE({base + "x.npy", base + "w.npy", base + "out.npy"});

    const Tensor x = oracle_npy(base + "x.npy").to_tensor();
    const Tensor w = oracle_npy(base + "w.npy").to_tensor();
    const Tensor expected = oracle_npy(base + "out.npy").to_tensor();

    Tensor actual = Tensor::zeros(expected.shape());
    ref::matmul(x, w, actual);
    assert_allclose(actual, expected, tol::kMatmul,
                    "matmul " + variant + " " + x.shape().to_string() + " x " +
                        w.shape().to_string());
  }
}

TEST_CASE("ref::matmul computes x times w transposed") {
  // x [2, 3], w [2, 3]: out[i][j] = dot(x_row_i, w_row_j).
  Tensor x = Tensor::zeros({2, 3});
  Tensor w = Tensor::zeros({2, 3});
  const float x_data[6] = {1, 2, 3, 4, 5, 6};
  const float w_data[6] = {1, 0, 0, 0, 1, 1};
  for (int i = 0; i < 6; ++i) {
    x.f32()[i] = x_data[i];
    w.f32()[i] = w_data[i];
  }

  Tensor out = Tensor::zeros({2, 2});
  ref::matmul(x, w, out);
  CHECK(out.f32()[0] == 1.0f);  // dot([1,2,3], [1,0,0])
  CHECK(out.f32()[1] == 5.0f);  // dot([1,2,3], [0,1,1])
  CHECK(out.f32()[2] == 4.0f);  // dot([4,5,6], [1,0,0])
  CHECK(out.f32()[3] == 11.0f); // dot([4,5,6], [0,1,1])
}

TEST_CASE("ref::matmul: identity weights reproduce the input") {
  const std::int64_t k = 7;
  Tensor x = random_tensor({3, k}, 11);
  Tensor w = Tensor::zeros({k, k});
  for (std::int64_t i = 0; i < k; ++i) w.f32()[i * k + i] = 1.0f;

  Tensor out = Tensor::zeros({3, k});
  ref::matmul(x, w, out);
  for (std::int64_t i = 0; i < x.numel(); ++i) CHECK(out.f32()[i] == x.f32()[i]);
}

TEST_CASE("ref::matmul rejects mismatched shapes") {
  const Tensor x = Tensor::zeros({2, 3});
  Tensor out = Tensor::zeros({2, 4});

  // w's second dimension is in_features and must match x's width.
  CHECK_THROWS_AS(ref::matmul(x, Tensor::zeros({4, 5}), out), std::invalid_argument);
  // out must be [m, n].
  CHECK_THROWS_AS(ref::matmul(x, Tensor::zeros({5, 3}), out), std::invalid_argument);
  // rank 3 is not a matrix.
  CHECK_THROWS_AS(ref::matmul(Tensor::zeros({2, 2, 2}), Tensor::zeros({2, 2}), out),
                  std::invalid_argument);
}
