// Workflow 06-attention-kv owns this file and test_kv_cache.cpp.
//
// In Phase 0 ops::attention forwards to ref::attention. When you optimize,
// the ref:: comparison covers arbitrary shapes cheaply and the oracle files
// pin the result to PyTorch. oracle/data/attention/ additionally holds the
// q/k/v projections and post-RoPE tensors for two real layers, so a whole
// attention block can be checked stage by stage.

#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <vector>

#include "common/allclose.h"
#include "common/oracle.h"
#include "common/random.h"
#include "minillm/kv_cache.h"
#include "minillm/ops.h"

using namespace minillm;
using namespace minillm::test;

TEST_CASE("ops::attention agrees with ref::attention at the model's shapes") {
  const float scale = 1.0f / 8.0f;  // 1 / sqrt(64)
  struct Case {
    const char* name;
    std::int64_t n_tokens;
    std::int64_t start_pos;
    std::int64_t n_kv;
  };
  const Case cases[] = {
      {"prefill 8", 8, 0, 8},
      {"decode at 8", 1, 8, 9},
      {"decode at 127", 1, 127, 128},
      {"continuation prefill", 4, 16, 20},
  };

  for (const Case& c : cases) {
    const Tensor q = random_tensor({c.n_tokens, 9, 64}, 401);
    const Tensor k = random_tensor({c.n_kv, 3, 64}, 402);
    const Tensor v = random_tensor({c.n_kv, 3, 64}, 403);
    Tensor expected = Tensor::zeros({c.n_tokens, 9, 64});
    Tensor actual = Tensor::zeros({c.n_tokens, 9, 64});

    ref::attention(q, k, v, c.start_pos, scale, expected);
    ops::attention(q, k, v, c.start_pos, scale, actual);
    assert_allclose(actual, expected, tol::kAttention,
                    std::string("ops::attention vs ref ") + c.name);
  }
}

TEST_CASE("ops::attention reads through a KVCache view") {
  // The path the model actually takes: project into the cache, then attend
  // over the prefix the cache hands out.
  const std::int64_t n_layers = 2;
  const std::int64_t max_seq = 16;
  const std::int64_t n_kv_heads = 3;
  const std::int64_t head_dim = 64;
  KVCache cache(n_layers, max_seq, n_kv_heads, head_dim);

  const std::int64_t n_tokens = 5;
  const Tensor k_src = random_tensor({n_tokens, n_kv_heads, head_dim}, 404);
  const Tensor v_src = random_tensor({n_tokens, n_kv_heads, head_dim}, 405);
  for (std::int64_t t = 0; t < n_tokens; ++t) {
    cache.store(1, t, k_src.f32() + t * n_kv_heads * head_dim,
                v_src.f32() + t * n_kv_heads * head_dim);
  }
  cache.set_seq_len(n_tokens);

  const Tensor q = random_tensor({n_tokens, 9, head_dim}, 406);
  Tensor expected = Tensor::zeros({n_tokens, 9, head_dim});
  Tensor actual = Tensor::zeros({n_tokens, 9, head_dim});

  ref::attention(q, k_src, v_src, 0, 0.125f, expected);
  ops::attention(q, cache.k(1, cache.seq_len()), cache.v(1, cache.seq_len()), 0, 0.125f, actual);
  assert_allclose(actual, expected, tol::kAttention, "ops::attention over a KVCache view");
}

TEST_CASE("ops::attention matches torch") {
  for (const std::string variant : {"small", "model_prefill", "model_decode"}) {
    const std::string base = "ops/attention_" + variant + "_";
    MINILLM_SKIP_WITHOUT_ORACLE({base + "q.npy", base + "k.npy", base + "v.npy",
                                 base + "start_pos.npy", base + "scale.npy", base + "out.npy"});

    const Tensor q = oracle_npy(base + "q.npy").to_tensor();
    const Tensor k = oracle_npy(base + "k.npy").to_tensor();
    const Tensor v = oracle_npy(base + "v.npy").to_tensor();
    const std::vector<std::int64_t> start_pos = oracle_npy(base + "start_pos.npy").as_i64();
    const std::vector<float> scale = oracle_npy(base + "scale.npy").as_f32();
    const Tensor expected = oracle_npy(base + "out.npy").to_tensor();
    REQUIRE(start_pos.size() == 1);
    REQUIRE(scale.size() == 1);

    Tensor actual = Tensor::zeros(expected.shape());
    ops::attention(q, k, v, start_pos[0], scale[0], actual);
    assert_allclose(actual, expected, tol::kAttention, "ops::attention " + variant);
  }
}

TEST_CASE("attention output for a real layer matches transformers") {
  // Stage-by-stage check of one real attention block: feed the golden
  // post-RoPE q/k and v_proj straight into the op and compare the output
  // before o_proj. This isolates the op from the projections.
  for (const std::string layer : {"l0", "l15"}) {
    const std::string base = "attention/" + layer + "_";
    MINILLM_SKIP_WITHOUT_ORACLE({base + "q_rope.npy", base + "k_rope.npy", base + "v_proj.npy",
                                 base + "attn_out.npy"});

    const Tensor q = oracle_npy(base + "q_rope.npy").to_tensor();
    const Tensor k = oracle_npy(base + "k_rope.npy").to_tensor();
    const Tensor v_flat = oracle_npy(base + "v_proj.npy").to_tensor();
    const Tensor expected = oracle_npy(base + "attn_out.npy").to_tensor();

    // v_proj is [n_tokens, n_kv_heads * head_dim]; the op wants it split.
    const Tensor v = v_flat.reshape({k.shape()[0], k.shape()[1], k.shape()[2]});
    const float scale = 1.0f / std::sqrt(static_cast<float>(q.shape()[2]));

    Tensor actual = Tensor::zeros(expected.shape());
    ops::attention(q, k, v, 0, scale, actual);
    assert_allclose(actual, expected, tol::kAttention, "attention block " + layer);
  }
}
