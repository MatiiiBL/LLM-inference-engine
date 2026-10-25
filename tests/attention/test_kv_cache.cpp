// Workflow 06-attention-kv owns this file.
//
// KVCache is implemented in Phase 0, so these are real tests. They pin the
// layout ([layer][position][kv_head][head_dim], separate K and V) and the
// "store does not advance seq_len" contract — both things an optimization
// pass is likely to want to change, and both things the model relies on.

#include <doctest/doctest.h>

#include <stdexcept>
#include <vector>

#include "common/random.h"
#include "minillm/kv_cache.h"

using namespace minillm;
using namespace minillm::test;

TEST_CASE("KVCache reports its geometry and allocates both buffers") {
  KVCache cache(4, 32, 3, 64);
  CHECK(cache.n_layers() == 4);
  CHECK(cache.max_seq_len() == 32);
  CHECK(cache.n_kv_heads() == 3);
  CHECK(cache.head_dim() == 64);
  CHECK(cache.seq_len() == 0);
  // Two buffers of 4 * 32 * 3 * 64 floats.
  CHECK(cache.bytes() == 2u * 4 * 32 * 3 * 64 * sizeof(float));

  CHECK_THROWS_AS(KVCache(0, 32, 3, 64), std::invalid_argument);
  CHECK_THROWS_AS(KVCache(4, 0, 3, 64), std::invalid_argument);
  CHECK_THROWS_AS(KVCache(4, 32, 0, 64), std::invalid_argument);
  CHECK_THROWS_AS(KVCache(4, 32, 3, 0), std::invalid_argument);
}

TEST_CASE("KVCache::for_model takes its sizes from the config") {
  ModelConfig cfg;
  cfg.num_hidden_layers = 30;
  cfg.num_key_value_heads = 3;
  cfg.head_dim = 64;
  cfg.max_position_embeddings = 8192;

  const KVCache cache = KVCache::for_model(cfg, 128);
  CHECK(cache.n_layers() == 30);
  CHECK(cache.max_seq_len() == 128);
  CHECK(cache.n_kv_heads() == 3);
  CHECK(cache.head_dim() == 64);

  CHECK_THROWS_AS(KVCache::for_model(cfg, 0), std::invalid_argument);
  CHECK_THROWS_AS(KVCache::for_model(cfg, 8193), std::invalid_argument);
}

TEST_CASE("KVCache stores keys and values separately, per layer and position") {
  const std::int64_t n_kv_heads = 2;
  const std::int64_t head_dim = 3;
  const std::int64_t slot = n_kv_heads * head_dim;
  KVCache cache(2, 4, n_kv_heads, head_dim);

  std::vector<float> k(static_cast<std::size_t>(slot));
  std::vector<float> v(static_cast<std::size_t>(slot));
  for (std::int64_t i = 0; i < slot; ++i) {
    k[static_cast<std::size_t>(i)] = static_cast<float>(100 + i);
    v[static_cast<std::size_t>(i)] = static_cast<float>(200 + i);
  }
  cache.store(1, 2, k.data(), v.data());
  cache.set_seq_len(3);

  const Tensor k_view = cache.k(1, 3);
  const Tensor v_view = cache.v(1, 3);
  REQUIRE(k_view.shape() == Shape{3, n_kv_heads, head_dim});
  CHECK(k_view.is_contiguous());

  // Position 2 holds what we wrote; positions 0 and 1 are still zero.
  for (std::int64_t h = 0; h < n_kv_heads; ++h) {
    for (std::int64_t d = 0; d < head_dim; ++d) {
      CHECK(k_view.f32()[k_view.offset(2, h, d)] == static_cast<float>(100 + h * head_dim + d));
      CHECK(v_view.f32()[v_view.offset(2, h, d)] == static_cast<float>(200 + h * head_dim + d));
      CHECK(k_view.f32()[k_view.offset(0, h, d)] == 0.0f);
    }
  }

  // Layer 0 is untouched: K and V are indexed by layer, not shared.
  const Tensor other_layer = cache.k(0, 3);
  for (std::int64_t i = 0; i < other_layer.numel(); ++i) CHECK(other_layer.f32()[i] == 0.0f);
}

TEST_CASE("KVCache slots are contiguous and writable in place") {
  KVCache cache(1, 4, 2, 3);
  float* k = cache.k_slot(0, 1);
  float* v = cache.v_slot(0, 1);
  CHECK(k != v);  // separate buffers

  for (std::int64_t i = 0; i < 6; ++i) {
    k[i] = static_cast<float>(i);
    v[i] = static_cast<float>(-i);
  }
  const Tensor k_view = cache.k(0, 2);
  CHECK(k_view.f32()[k_view.offset(1, 0, 0)] == 0.0f);
  CHECK(k_view.f32()[k_view.offset(1, 1, 2)] == 5.0f);

  // Consecutive positions are one slot apart: the layout is [pos][head][dim].
  CHECK(cache.k_slot(0, 2) - cache.k_slot(0, 1) == 6);
  // Consecutive layers are max_seq_len slots apart.
  KVCache two_layers(2, 4, 2, 3);
  CHECK(two_layers.k_slot(1, 0) - two_layers.k_slot(0, 0) == 4 * 6);
}

TEST_CASE("KVCache::store does not advance seq_len") {
  KVCache cache(1, 4, 1, 2);
  const std::vector<float> data{1.0f, 2.0f};
  cache.store(0, 0, data.data(), data.data());
  CHECK(cache.seq_len() == 0);  // the model calls set_seq_len() when the pass is done
  cache.set_seq_len(1);
  CHECK(cache.seq_len() == 1);
}

TEST_CASE("KVCache::reset forgets positions without clearing memory") {
  KVCache cache(1, 4, 1, 2);
  const std::vector<float> data{1.0f, 2.0f};
  cache.store(0, 0, data.data(), data.data());
  cache.set_seq_len(1);

  cache.reset();
  CHECK(cache.seq_len() == 0);
  // The bytes are still there — correctness comes from never reading past the
  // length passed to k()/v(), not from zeroing.
  CHECK(cache.k(0, 1).f32()[0] == 1.0f);
}

TEST_CASE("KVCache bounds-checks every access") {
  KVCache cache(2, 4, 1, 2);
  const std::vector<float> data{1.0f, 2.0f};

  CHECK_THROWS_AS(cache.store(2, 0, data.data(), data.data()), std::out_of_range);
  CHECK_THROWS_AS(cache.store(-1, 0, data.data(), data.data()), std::out_of_range);
  CHECK_THROWS_AS(cache.store(0, 4, data.data(), data.data()), std::out_of_range);
  CHECK_THROWS_AS(cache.store(0, -1, data.data(), data.data()), std::out_of_range);
  CHECK_THROWS_AS(cache.store(0, 0, nullptr, data.data()), std::invalid_argument);

  CHECK_THROWS_AS(cache.k_slot(2, 0), std::out_of_range);
  CHECK_THROWS_AS(cache.v_slot(0, 4), std::out_of_range);
  CHECK_THROWS_AS(cache.k(2, 1), std::out_of_range);
  CHECK_THROWS_AS(cache.v(0, 5), std::out_of_range);
  CHECK_THROWS_AS(cache.set_seq_len(5), std::out_of_range);
  CHECK_THROWS_AS(cache.set_seq_len(-1), std::out_of_range);

  // A zero-length view is legal: it is what the first prefill sees.
  CHECK(cache.k(0, 0).shape() == Shape{0, 1, 2});
}

TEST_CASE("a default-constructed KVCache is empty") {
  const KVCache cache;
  CHECK(cache.n_layers() == 0);
  CHECK(cache.max_seq_len() == 0);
  CHECK(cache.seq_len() == 0);
  CHECK(cache.bytes() == 0);
}
