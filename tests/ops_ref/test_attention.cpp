// ref::attention vs PyTorch eager attention.
//
// Three things are easy to get wrong here and all three are tested: the causal
// bound (query t sees keys 0..start_pos+t), the GQA mapping (query head h uses
// KV head h / group_size) and the scale.

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

TEST_CASE("ref::attention matches torch") {
  // prefill: 8 queries from an empty cache. decode: 1 query against 16 cached
  // positions — the two shapes the engine actually runs.
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
    ref::attention(q, k, v, start_pos[0], scale[0], actual);
    assert_allclose(actual, expected, tol::kAttention,
                    "attention " + variant + " q" + q.shape().to_string() + " kv" +
                        k.shape().to_string() + " start_pos=" + std::to_string(start_pos[0]));
  }
}

TEST_CASE("ref::attention is causal") {
  // One head, head_dim 1, three positions. V is 1, 2, 3 and all queries and
  // keys are zero, so every score is equal and the output at position t is the
  // mean of the first t+1 values.
  const std::int64_t n = 3;
  Tensor q = Tensor::zeros({n, 1, 1});
  Tensor k = Tensor::zeros({n, 1, 1});
  Tensor v = Tensor::zeros({n, 1, 1});
  for (std::int64_t i = 0; i < n; ++i) v.f32()[i] = static_cast<float>(i + 1);

  Tensor out = Tensor::zeros({n, 1, 1});
  ref::attention(q, k, v, 0, 1.0f, out);

  CHECK(out.f32()[0] == doctest::Approx(1.0f));                  // mean(1)
  CHECK(out.f32()[1] == doctest::Approx(1.5f));                  // mean(1, 2)
  CHECK(out.f32()[2] == doctest::Approx(2.0f));                  // mean(1, 2, 3)
}

TEST_CASE("ref::attention with start_pos sees the whole cached prefix") {
  // One query at absolute position 2, cache holding positions 0..2.
  Tensor q = Tensor::zeros({1, 1, 1});
  Tensor k = Tensor::zeros({3, 1, 1});
  Tensor v = Tensor::zeros({3, 1, 1});
  for (std::int64_t i = 0; i < 3; ++i) v.f32()[i] = static_cast<float>(i + 1);

  Tensor out = Tensor::zeros({1, 1, 1});
  ref::attention(q, k, v, 2, 1.0f, out);
  CHECK(out.f32()[0] == doctest::Approx(2.0f));  // mean(1, 2, 3)

  // A longer cache than needed is allowed; the extra positions are ignored.
  Tensor long_k = Tensor::zeros({8, 1, 1});
  Tensor long_v = Tensor::zeros({8, 1, 1});
  for (std::int64_t i = 0; i < 8; ++i) long_v.f32()[i] = static_cast<float>(i + 1);
  ref::attention(q, long_k, long_v, 2, 1.0f, out);
  CHECK(out.f32()[0] == doctest::Approx(2.0f));  // still only positions 0..2
}

TEST_CASE("ref::attention maps query heads onto KV heads by integer division") {
  // 4 query heads, 2 KV heads => group size 2: heads 0,1 -> KV 0; 2,3 -> KV 1.
  // Queries and keys are zero so attention is uniform; V differs per KV head,
  // so the output reveals which KV head each query head read.
  const std::int64_t head_dim = 2;
  Tensor q = Tensor::zeros({1, 4, head_dim});
  Tensor k = Tensor::zeros({1, 2, head_dim});
  Tensor v = Tensor::zeros({1, 2, head_dim});
  v.f32()[0] = 10.0f;  // KV head 0, dim 0
  v.f32()[1] = 11.0f;  // KV head 0, dim 1
  v.f32()[2] = 20.0f;  // KV head 1, dim 0
  v.f32()[3] = 21.0f;  // KV head 1, dim 1

  Tensor out = Tensor::zeros({1, 4, head_dim});
  ref::attention(q, k, v, 0, 1.0f, out);

  CHECK(out.f32()[0] == doctest::Approx(10.0f));
  CHECK(out.f32()[1] == doctest::Approx(11.0f));
  CHECK(out.f32()[2] == doctest::Approx(10.0f));
  CHECK(out.f32()[3] == doctest::Approx(11.0f));
  CHECK(out.f32()[4] == doctest::Approx(20.0f));
  CHECK(out.f32()[5] == doctest::Approx(21.0f));
  CHECK(out.f32()[6] == doctest::Approx(20.0f));
  CHECK(out.f32()[7] == doctest::Approx(21.0f));
}

TEST_CASE("ref::attention applies the scale to the logits") {
  // Two keys, one query. The score difference is scaled, so the softmax
  // weights follow sigmoid(scale * delta).
  Tensor q = Tensor::zeros({1, 1, 1});
  Tensor k = Tensor::zeros({2, 1, 1});
  Tensor v = Tensor::zeros({2, 1, 1});
  q.f32()[0] = 1.0f;
  k.f32()[0] = 0.0f;
  k.f32()[1] = 1.0f;
  v.f32()[0] = 0.0f;
  v.f32()[1] = 1.0f;

  Tensor out = Tensor::zeros({1, 1, 1});
  ref::attention(q, k, v, 1, 2.0f, out);
  // scores are [0, 2]; weight on v[1] is exp(2)/(1+exp(2)).
  const float want = std::exp(2.0f) / (1.0f + std::exp(2.0f));
  CHECK(out.f32()[0] == doctest::Approx(want).epsilon(1e-6));
}

TEST_CASE("ref::attention accepts a KV cache view") {
  // The contiguous case is what KVCache::k(layer, length) hands out: one
  // layer's prefix of a [layer, pos, kv_head, head_dim] buffer.
  Tensor cache = Tensor::zeros({2, 4, 1, 1});
  for (std::int64_t i = 0; i < cache.numel(); ++i) cache.f32()[i] = static_cast<float>(i);

  const Tensor prefix = cache.row(1).slice(0, 0, 2);
  CHECK(prefix.shape() == Shape{2, 1, 1});
  CHECK(prefix.is_contiguous());

  Tensor q = Tensor::zeros({1, 1, 1});
  Tensor out = Tensor::zeros({1, 1, 1});
  ref::attention(q, prefix, prefix, 1, 1.0f, out);
  // Layer 1 holds values 4, 5, 6, 7; the mean of the first two is 4.5.
  CHECK(out.f32()[0] == doctest::Approx(4.5f));

  // A genuinely strided view also works, because indexing goes through
  // Tensor::offset() rather than assuming packed strides.
  Tensor two_heads = Tensor::zeros({1, 4, 2, 1});
  for (std::int64_t i = 0; i < two_heads.numel(); ++i) {
    two_heads.f32()[i] = static_cast<float>(i);
  }
  const Tensor head0 = two_heads.row(0).slice(1, 0, 1);
  CHECK(head0.shape() == Shape{4, 1, 1});
  CHECK_FALSE(head0.is_contiguous());

  ref::attention(q, head0, head0, 1, 1.0f, out);
  // KV head 0 holds values 0, 2, 4, 6; the mean of the first two is 1.
  CHECK(out.f32()[0] == doctest::Approx(1.0f));
}

TEST_CASE("ref::attention validates its arguments") {
  const Tensor q = Tensor::zeros({2, 4, 8});
  const Tensor k = Tensor::zeros({4, 2, 8});
  Tensor out = Tensor::zeros({2, 4, 8});

  // Cache too short for start_pos + n_tokens.
  CHECK_THROWS_AS(ref::attention(q, k, k, 3, 1.0f, out), std::invalid_argument);
  // n_q_heads not a multiple of n_kv_heads.
  const Tensor k3 = Tensor::zeros({4, 3, 8});
  CHECK_THROWS_AS(ref::attention(q, k3, k3, 0, 1.0f, out), std::invalid_argument);
  // head_dim mismatch between q and the cache.
  const Tensor k_wide = Tensor::zeros({4, 2, 16});
  CHECK_THROWS_AS(ref::attention(q, k_wide, k_wide, 0, 1.0f, out), std::invalid_argument);
  // out shape must equal q shape.
  Tensor out_wrong = Tensor::zeros({2, 4, 4});
  CHECK_THROWS_AS(ref::attention(q, k, k, 0, 1.0f, out_wrong), std::invalid_argument);
  // negative start_pos.
  CHECK_THROWS_AS(ref::attention(q, k, k, -1, 1.0f, out), std::invalid_argument);
}
