// The forward pass is assembled after Wave 1; this file comes with it.
//
// PLACEHOLDER. Model is a stub.
//
// What this must cover when it is done, against oracle/data/model/:
//   * for each of the 3 prompts: every hidden state p{i}_hidden_{j}.npy
//     compared layer by layer via the hidden-state hook, using
//     tol::kHiddenState — and the first layer that drifts is the one to debug,
//     so check them in order and report the first failure
//   * final logits vs p{i}_logits.npy using tol::kLogits
//   * the 64-token greedy generation vs p{i}_greedy_ids.npy compared EXACTLY
//     with assert_equal — one divergent token is a real bug
//   * prefill-then-decode equals one-shot prefill: forward(prompt, 0) then
//     forward(next, n) must give the same logits as forward(prompt+next, 0)
//   * reset() lets the same Model run a second prompt from position 0
//   * the hook index mapping matches transformers: index 0 is the embedding
//     output, index num_hidden_layers is post-final-norm
//
// Also the place to measure the real per-layer drift and replace the
// PROVISIONAL tolerances in tests/common/tolerances.h with measured values.

#include <doctest/doctest.h>

#include <stdexcept>
#include <vector>

#include "common/allclose.h"
#include "common/oracle.h"
#include "minillm/model.h"

using namespace minillm;
using namespace minillm::test;

TEST_CASE("model stubs are linked and announce themselves") {
  MESSAGE("SKIP: the forward pass is a stub — it is assembled after Wave 1");
  note_skip("model: forward-pass tests pending the post-Wave-1 assembly");

  CHECK_THROWS_AS(Model(ModelConfig{}, Weights{}, 128), std::logic_error);
  CHECK_THROWS_AS(Model::from_dir(models_dir(), 128), std::logic_error);
}

TEST_CASE("the model oracle data is where the tests expect it") {
  MINILLM_SKIP_WITHOUT_ORACLE({"model/prompts.json", "model/p0_input_ids.npy",
                               "model/p0_hidden_0.npy", "model/p0_logits.npy",
                               "model/p0_greedy_ids.npy"});
  // The shapes are worth checking even before the model exists: they encode
  // the hook contract (num_hidden_layers + 1 hidden states per prompt).
  const NpyArray ids = oracle_npy("model/p0_input_ids.npy");
  const NpyArray hidden = oracle_npy("model/p0_hidden_0.npy");
  const NpyArray logits = oracle_npy("model/p0_logits.npy");
  const NpyArray greedy = oracle_npy("model/p0_greedy_ids.npy");

  REQUIRE(ids.shape().size() == 1);
  REQUIRE(hidden.shape().size() == 2);
  CHECK(hidden.shape()[0] == ids.shape()[0]);
  REQUIRE(logits.shape().size() == 2);
  CHECK(logits.shape()[0] == ids.shape()[0]);
  CHECK(greedy.shape().size() == 1);
  CHECK(greedy.shape()[0] == 64);
}
