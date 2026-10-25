// Workflow 01-loader owns this file.
//
// PLACEHOLDER. Everything that parses is a stub, so the only real assertions
// here are that the stubs are wired up and that the Weights container works.
// Replace the stub checks with real ones as you implement; the oracle files
// each test needs are named in its skip message.
//
// What this must cover when it is done:
//   * ModelConfig parsed from the real config.json: every field, head_dim
//     derived when absent, rope_scaling rejected, validate() catching a
//     head count that does not divide
//   * safetensors header parsing: offsets, stored dtypes, a malformed header
//   * bf16 -> fp32 conversion being exact
//   * every expected tensor name present with the right shape, and
//     lm_head.weight absent because tie_word_embeddings is true
//   * the weights/samples.json spot checks: first 16 values and fp32 sum

#include <doctest/doctest.h>

#include <stdexcept>
#include <vector>

#include "common/allclose.h"
#include "common/oracle.h"
#include "minillm/config.h"
#include "minillm/loader.h"

using namespace minillm;
using namespace minillm::test;

TEST_CASE("Weights is a name -> tensor map with useful errors") {
  Weights w;
  CHECK(w.size() == 0);
  CHECK_FALSE(w.has("model.embed_tokens.weight"));
  CHECK(w.find("nope") == nullptr);
  CHECK_THROWS_AS(w.get("nope"), std::runtime_error);

  w.insert("model.norm.weight", Tensor::zeros({576}));
  w.insert("model.embed_tokens.weight", Tensor::zeros({4, 8}));
  CHECK(w.size() == 2);
  CHECK(w.has("model.norm.weight"));
  CHECK(w.get("model.norm.weight").shape() == Shape{576});
  CHECK(w.bytes() == (576 + 32) * sizeof(float));

  // names() is sorted, so tests and error messages are stable.
  const std::vector<std::string> names = w.names();
  REQUIRE(names.size() == 2);
  CHECK(names[0] == "model.embed_tokens.weight");
  CHECK(names[1] == "model.norm.weight");

  CHECK_THROWS_AS(w.insert("model.norm.weight", Tensor::zeros({1})), std::runtime_error);
}

TEST_CASE("ModelConfig derived quantities are computed, not stored") {
  ModelConfig cfg;
  cfg.num_attention_heads = 9;
  cfg.num_key_value_heads = 3;
  cfg.head_dim = 64;
  CHECK(cfg.kv_group_size() == 3);
  CHECK(cfg.q_dim() == 576);
  CHECK(cfg.kv_dim() == 192);
  CHECK(cfg.attention_scale() == doctest::Approx(0.125f));

  ModelConfig bad;
  CHECK_THROWS_AS(bad.attention_scale(), std::invalid_argument);
}

TEST_CASE("loader stubs are linked and announce themselves") {
  MESSAGE("SKIP: the loader is a stub — workflow 01-loader implements it");
  note_skip("loader: every parsing test is pending workflow 01-loader");

  CHECK_THROWS_AS(ModelConfig::from_json_file("config.json"), std::logic_error);
  CHECK_THROWS_AS(ModelConfig::from_json_string("{}"), std::logic_error);
  CHECK_THROWS_AS(ModelConfig().validate(), std::logic_error);
  CHECK_THROWS_AS(convert_to_f32(Tensor::zeros({2}, DType::BF16)), std::logic_error);
  CHECK_THROWS_AS(load_safetensors("model.safetensors"), std::logic_error);
  CHECK_THROWS_AS(load_checkpoint("."), std::logic_error);
  CHECK_THROWS_AS(read_safetensors_header("model.safetensors"), std::logic_error);
}
