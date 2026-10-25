// Workflow 04-matmul owns this file.
//
// In Phase 0 ops::matmul forwards to ref::matmul, so these pass trivially.
// They become the real safety net the moment you write a fast kernel: keep
// both the ref:: comparison (cheap, any shape) and the torch comparison
// (ground truth), and add the shapes your kernel special-cases.

#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "common/allclose.h"
#include "common/oracle.h"
#include "common/random.h"
#include "minillm/ops.h"

using namespace minillm;
using namespace minillm::test;

TEST_CASE("ops::matmul agrees with ref::matmul at the model's shapes") {
  // [m, k] x [n, k] for every linear layer in the forward pass. m = 1 is the
  // decode step, m = 8 a short prefill.
  struct Case {
    const char* name;
    std::int64_t m;
    std::int64_t k;
    std::int64_t n;
  };
  const Case cases[] = {
      {"q_proj decode", 1, 576, 576},   {"q_proj prefill", 8, 576, 576},
      {"kv_proj prefill", 8, 576, 192}, {"mlp_up prefill", 8, 576, 1536},
      {"mlp_down prefill", 8, 1536, 576}, {"lm_head decode", 1, 576, 4096},
  };

  for (const Case& c : cases) {
    const Tensor x = random_tensor({c.m, c.k}, 201);
    const Tensor w = random_tensor({c.n, c.k}, 202);
    Tensor expected = Tensor::zeros({c.m, c.n});
    Tensor actual = Tensor::zeros({c.m, c.n});

    ref::matmul(x, w, expected);
    ops::matmul(x, w, actual);
    assert_allclose(actual, expected, tol::kMatmul, std::string("ops::matmul vs ref ") + c.name);
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
