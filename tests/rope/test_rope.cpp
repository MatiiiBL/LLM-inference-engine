// Workflow 05-rope owns this file.
//
// In Phase 0 ops::rope forwards to ref::rope. The comparisons that matter when
// you optimize: cached cos/sin tables must agree with recomputed values, and
// the rotate_half pairing must survive whatever reindexing you do.

#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "common/allclose.h"
#include "common/oracle.h"
#include "common/random.h"
#include "minillm/ops.h"

using namespace minillm;
using namespace minillm::test;

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
