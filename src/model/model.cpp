// Model — the forward pass, assembled from the loader, the ops and the KV cache.
//
// The data flow is docs/ARCHITECTURE.md §4, step for step. The constructor
// does every check that depends on the checkpoint, so forward() checks only
// its own arguments and then runs straight through.
//
// Memory: the weights are held by handle (Tensor copies share storage, so
// nothing is copied, including the tied LM head). The projections and the
// embedding may be Q8_0 (quantize_weights_q8_0); forward() never looks at a
// dtype, because ops::matmul and ops::embedding dispatch on it. The KV cache
// and one Arena are allocated in the constructor, both sized for
// max_seq_len; forward() rewinds the arena and carves its buffers from it
// (src/model/scratch.h), so a pass never touches the heap.
//
// Keys and values are projected straight into the cache: positions
// [start_pos, start_pos + n) of one layer are a contiguous [n, kv_dim] block,
// so the k_proj/v_proj matmuls write there and RoPE rotates the keys in place.
// The cache therefore always holds post-RoPE keys, and store() is never
// needed.
//
// Single-threaded on purpose: any parallelism lives inside the ops.

#include <cstddef>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "minillm/model.h"
#include "minillm/ops.h"
#include "minillm/runtime.h"
#include "scratch.h"

namespace minillm {
namespace {

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("Model: " + message);
}

[[noreturn]] void bad_argument(const std::string& message) {
  throw std::invalid_argument("Model: " + message);
}

/// validate(), plus the one thing ops::rope would otherwise only reject on
/// the first forward(): RoPE rotates pairs, so head_dim must be even.
void check_config(const ModelConfig& cfg) {
  cfg.validate();
  if (cfg.head_dim % 2 != 0) {
    fail("head_dim must be even for RoPE, got " + std::to_string(cfg.head_dim));
  }
}

/// max_seq_len capped at the model's context, as model.h promises.
std::int64_t checked_max_seq_len(const ModelConfig& cfg, std::int64_t max_seq_len) {
  if (max_seq_len <= 0) {
    bad_argument("max_seq_len must be positive, got " + std::to_string(max_seq_len));
  }
  return max_seq_len < cfg.max_position_embeddings ? max_seq_len : cfg.max_position_embeddings;
}

/// What a tensor may be stored as: norms are always F32, while the
/// projections, the embedding and an untied LM head may also be Q8_0.
enum class Stored { kF32, kF32OrQ8_0 };

/// Looks tensors up by name and checks each against the shape `cfg` implies,
/// remembering which names it has handed out so that leftovers can be
/// reported afterwards.
class TensorBinder {
 public:
  explicit TensorBinder(const Weights& weights) : weights_(weights) {}

  /// The tensor `name`, which must exist with exactly `shape`, stored as
  /// `stored` allows.
  const Tensor& require(const std::string& name, const Shape& shape,
                        Stored stored = Stored::kF32) {
    const Tensor* t = weights_.find(name);
    if (t == nullptr) {
      fail("missing tensor '" + name + "': expected f32" + shape.to_string() +
           ", the checkpoint has none");
    }
    check(name, *t, shape, stored);
    used_.insert(name);
    return *t;
  }

  /// Throws naming the first tensor that no require() call asked for.
  void reject_unused() const {
    if (weights_.size() == used_.size()) return;
    for (const std::string& name : weights_.names()) {
      if (used_.count(name) == 0) {
        fail("unexpected tensor '" + name + "' (" + weights_.get(name).to_string() + "): " +
             std::to_string(weights_.size() - used_.size()) +
             " tensor(s) in the checkpoint are not part of the model this config describes");
      }
    }
  }

  void mark_used(const std::string& name) { used_.insert(name); }

  /// A misshapen tensor of an allowed dtype is reported against the same
  /// dtype; a disallowed dtype against every dtype the slot takes.
  static void check(const std::string& name, const Tensor& t, const Shape& shape, Stored stored) {
    const bool dtype_ok =
        t.dtype() == DType::F32 || (stored == Stored::kF32OrQ8_0 && t.dtype() == DType::Q8_0);
    if (!dtype_ok || t.shape() != shape) {
      const char* expected = dtype_ok                       ? dtype_name(t.dtype())
                             : stored == Stored::kF32OrQ8_0 ? "f32 or q8_0"
                                                            : "f32";
      fail("tensor '" + name + "' is " + dtype_name(t.dtype()) + t.shape().to_string() +
           ", expected " + expected + shape.to_string());
    }
    if (!t.is_contiguous()) fail("tensor '" + name + "' is not contiguous: " + t.to_string());
  }

 private:
  const Weights& weights_;
  std::set<std::string> used_;
};

/// One decoder layer's weights, by handle.
struct LayerWeights {
  Tensor input_norm;  ///< [hidden]
  Tensor q_proj;      ///< [n_heads * head_dim, hidden]
  Tensor k_proj;      ///< [n_kv_heads * head_dim, hidden]
  Tensor v_proj;      ///< [n_kv_heads * head_dim, hidden]
  Tensor o_proj;      ///< [hidden, n_heads * head_dim]
  Tensor post_norm;   ///< [hidden]
  Tensor gate_proj;   ///< [intermediate, hidden]
  Tensor up_proj;     ///< [intermediate, hidden]
  Tensor down_proj;   ///< [hidden, intermediate]
};

}  // namespace

struct Model::Impl {
  Impl(ModelConfig cfg, std::int64_t max_seq_len)
      : config(std::move(cfg)),
        cache(KVCache::for_model(config, max_seq_len)),
        arena(model_detail::scratch_bytes(config, max_seq_len)),
        attention_scale(config.attention_scale()) {}

  ModelConfig config;
  Tensor embed_tokens;  ///< [vocab, hidden]
  std::vector<LayerWeights> layers;
  Tensor final_norm;  ///< [hidden]
  Tensor lm_head;     ///< [vocab, hidden]; the embed_tokens handle itself when tied
  KVCache cache;
  Arena arena;
  float attention_scale;
  Model::HiddenStateHook hook;
};

Model::Model(ModelConfig cfg, Weights weights, std::int64_t max_seq_len) {
  check_config(cfg);
  const std::int64_t seq_len = checked_max_seq_len(cfg, max_seq_len);

  // Every tensor is checked before the cache and the arena are allocated, so
  // a bad checkpoint fails before claiming any more memory.
  const std::int64_t hidden = cfg.hidden_size;
  const std::int64_t inter = cfg.intermediate_size;
  constexpr Stored kWeight = Stored::kF32OrQ8_0;
  TensorBinder bind(weights);
  Tensor embed_tokens =
      bind.require("model.embed_tokens.weight", Shape{cfg.vocab_size, hidden}, kWeight);
  std::vector<LayerWeights> layers(static_cast<std::size_t>(cfg.num_hidden_layers));
  for (std::int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    LayerWeights& lw = layers[static_cast<std::size_t>(i)];
    lw.input_norm = bind.require(p + "input_layernorm.weight", Shape{hidden});
    lw.q_proj = bind.require(p + "self_attn.q_proj.weight", Shape{cfg.q_dim(), hidden}, kWeight);
    lw.k_proj = bind.require(p + "self_attn.k_proj.weight", Shape{cfg.kv_dim(), hidden}, kWeight);
    lw.v_proj = bind.require(p + "self_attn.v_proj.weight", Shape{cfg.kv_dim(), hidden}, kWeight);
    lw.o_proj = bind.require(p + "self_attn.o_proj.weight", Shape{hidden, cfg.q_dim()}, kWeight);
    lw.post_norm = bind.require(p + "post_attention_layernorm.weight", Shape{hidden});
    lw.gate_proj = bind.require(p + "mlp.gate_proj.weight", Shape{inter, hidden}, kWeight);
    lw.up_proj = bind.require(p + "mlp.up_proj.weight", Shape{inter, hidden}, kWeight);
    lw.down_proj = bind.require(p + "mlp.down_proj.weight", Shape{hidden, inter}, kWeight);
  }
  Tensor final_norm = bind.require("model.norm.weight", Shape{hidden});

  // Tied embeddings: the LM head IS embed_tokens — the same handle, never a
  // copy. A tied checkpoint may still carry an lm_head.weight (some exporters
  // write the shared matrix twice). transformers would quietly ignore it; we
  // accept it only if it is bit for bit embed_tokens, in the same dtype
  // (quantize_weights_q8_0 quantizes both, to the same bytes), because
  // anything else means the config and the checkpoint disagree about the
  // model, and picking either one silently computes plausible but wrong
  // logits. An accepted copy is dropped with the rest of `weights` when the
  // constructor returns.
  Tensor lm_head;
  const std::string lm_head_name = "lm_head.weight";
  if (!cfg.tie_word_embeddings) {
    lm_head = bind.require(lm_head_name, Shape{cfg.vocab_size, hidden}, kWeight);
  } else {
    lm_head = embed_tokens;
    if (const Tensor* extra = weights.find(lm_head_name)) {
      TensorBinder::check(lm_head_name, *extra, embed_tokens.shape(), kWeight);
      if (extra->dtype() != embed_tokens.dtype()) {
        fail("tie_word_embeddings is true, but the checkpoint's lm_head.weight is " +
             std::string(dtype_name(extra->dtype())) + " and model.embed_tokens.weight " +
             dtype_name(embed_tokens.dtype()));
      }
      if (extra->data() != embed_tokens.data() &&
          std::memcmp(extra->data(), embed_tokens.data(), embed_tokens.nbytes()) != 0) {
        fail("tie_word_embeddings is true, but the checkpoint's lm_head.weight differs from "
             "model.embed_tokens.weight; the config and the checkpoint disagree");
      }
      bind.mark_used(lm_head_name);
    }
  }

  // A tensor the model does not use is an error, not something to skip: the
  // likeliest cause is a config that describes a different model than the
  // checkpoint (a 32-layer checkpoint read as 30 layers runs, and produces
  // fluent garbage). That includes derived buffers some exporters save, such
  // as rotary_emb.inv_freq — this engine computes RoPE from rope_theta.
  bind.reject_unused();

  impl_ = std::make_unique<Impl>(std::move(cfg), seq_len);
  impl_->embed_tokens = std::move(embed_tokens);
  impl_->layers = std::move(layers);
  impl_->final_norm = std::move(final_norm);
  impl_->lm_head = std::move(lm_head);
}

Model::~Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

void convert_weights_bf16(Weights&, const ModelConfig&) {
  throw std::logic_error("not implemented: convert_weights_bf16 (workflow 21)");
}

void quantize_weights_q8_0(Weights& weights, const ModelConfig& cfg) {
  std::vector<std::string> names = {"model.embed_tokens.weight"};
  for (std::int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    for (const char* proj : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                             "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}) {
      names.push_back(p + proj + ".weight");
    }
  }
  // Untied, lm_head is required; tied, a stray copy is quantized too, so the
  // constructor still finds it equal to the quantized embed_tokens.
  if (!cfg.tie_word_embeddings || weights.has("lm_head.weight")) names.push_back("lm_head.weight");

  // A missing tensor is reported before anything is replaced.
  for (const std::string& name : names) weights.get(name);
  for (const std::string& name : names) {
    Tensor quantized;
    try {
      quantized = quantize_q8_0(weights.get(name));
    } catch (const std::invalid_argument& e) {
      throw std::invalid_argument("quantize_weights_q8_0: tensor '" + name + "': " + e.what());
    }
    // Drops the F32 storage now, so the peak stays at the F32 checkpoint plus
    // one Q8_0 tensor.
    weights.replace(name, std::move(quantized));
  }
}

Model Model::from_dir(const std::filesystem::path& model_dir, std::int64_t max_seq_len,
                      DType weight_dtype) {
  if (weight_dtype != DType::F32 && weight_dtype != DType::Q8_0) {
    bad_argument("from_dir: weight_dtype must be f32 or q8_0, got " +
                 std::string(dtype_name(weight_dtype)));
  }
  ModelConfig cfg = ModelConfig::from_json_file(model_dir / "config.json");
  check_config(cfg);
  // A bad argument should not wait on reading the weights.
  checked_max_seq_len(cfg, max_seq_len);
  Weights weights = load_checkpoint(model_dir);
  if (weight_dtype == DType::Q8_0) quantize_weights_q8_0(weights, cfg);
  return Model(std::move(cfg), std::move(weights), max_seq_len);
}

std::span<const float> Model::forward(std::span<const std::int32_t> tokens, std::int64_t start_pos) {
  Impl& m = *impl_;
  const ModelConfig& cfg = m.config;
  KVCache& cache = m.cache;

  // Every argument is checked before anything is written, so a rejected call
  // leaves the cache and the previous call's logits exactly as they were.
  const auto n = static_cast<std::int64_t>(tokens.size());
  if (n == 0) bad_argument("forward: tokens is empty");
  for (std::size_t t = 0; t < tokens.size(); ++t) {
    if (tokens[t] < 0 || tokens[t] >= cfg.vocab_size) {
      bad_argument("forward: token id " + std::to_string(tokens[t]) + " at index " +
                   std::to_string(t) + " is outside [0, " + std::to_string(cfg.vocab_size) + ")");
    }
  }
  if (start_pos < 0) bad_argument("forward: start_pos " + std::to_string(start_pos) + " is negative");
  // Attention reads every position before start_pos, so they must all have
  // been written. Starting earlier is fine: it overwrites from start_pos on.
  if (start_pos > cache.seq_len()) {
    bad_argument("forward: start_pos " + std::to_string(start_pos) + " is past the " +
                 std::to_string(cache.seq_len()) +
                 " cached positions, so the positions in between were never written");
  }
  if (n > cache.max_seq_len() - start_pos) {
    bad_argument("forward: start_pos " + std::to_string(start_pos) + " + " + std::to_string(n) +
                 " tokens exceeds the KV cache's " + std::to_string(cache.max_seq_len()) +
                 " positions");
  }

  m.arena.reset();
  model_detail::Scratch s = model_detail::carve(m.arena, cfg, n);
  for (std::int64_t t = 0; t < n; ++t) s.positions[t] = start_pos + t;
  const std::span<const std::int64_t> positions(s.positions, static_cast<std::size_t>(n));

  const std::int64_t len = start_pos + n;
  const std::int64_t n_layers = cfg.num_hidden_layers;
  const Shape kv_rows{n, cfg.kv_dim()};
  const Shape kv_heads{n, cfg.num_key_value_heads, cfg.head_dim};
  // matmul sees q and the attention output as [n, n_heads * head_dim] rows.
  Tensor q_rows = s.q.reshape(Shape{n, cfg.q_dim()});
  const Tensor attn_rows = s.attn.reshape(Shape{n, cfg.q_dim()});

  ops::embedding(m.embed_tokens, tokens, s.h);
  for (std::int64_t l = 0; l < n_layers; ++l) {
    // The input to layer l; for l = 0 that is the embedding output.
    if (m.hook) m.hook(l, s.h);
    const LayerWeights& lw = m.layers[static_cast<std::size_t>(l)];

    // ---- attention block ----
    ops::rmsnorm(s.h, lw.input_norm, cfg.rms_norm_eps, s.x);
    ops::matmul(s.x, lw.q_proj, q_rows);
    Tensor k_new = Tensor::wrap(cache.k_slot(l, start_pos), kv_rows);
    Tensor v_new = Tensor::wrap(cache.v_slot(l, start_pos), kv_rows);
    ops::matmul(s.x, lw.k_proj, k_new);
    ops::matmul(s.x, lw.v_proj, v_new);
    Tensor k_rot = k_new.reshape(kv_heads);
    ops::rope(s.q, &k_rot, positions, cfg.rope_theta);
    ops::attention(s.q, cache.k(l, len), cache.v(l, len), start_pos, m.attention_scale, s.attn);
    ops::matmul(attn_rows, lw.o_proj, s.x);
    ops::add(s.h, s.x, s.h);

    // ---- MLP block (SwiGLU) ----
    ops::rmsnorm(s.h, lw.post_norm, cfg.rms_norm_eps, s.x);
    ops::matmul(s.x, lw.gate_proj, s.gate);
    ops::matmul(s.x, lw.up_proj, s.up);
    ops::silu_mul(s.gate, s.up, s.gate);
    ops::matmul(s.gate, lw.down_proj, s.x);
    ops::add(s.h, s.x, s.h);
  }
  cache.set_seq_len(len);

  // Every row is normed, because the hook sees all of them; only the last
  // row goes through the LM head.
  ops::rmsnorm(s.h, m.final_norm, cfg.rms_norm_eps, s.h);
  if (m.hook) m.hook(n_layers, s.h);
  ops::matmul(s.h.slice(0, n - 1, 1), m.lm_head, s.logits);
  return {s.logits.f32(), static_cast<std::size_t>(cfg.vocab_size)};
}

const ModelConfig& Model::config() const { return impl_->config; }

KVCache& Model::kv_cache() { return impl_->cache; }

const KVCache& Model::kv_cache() const { return impl_->cache; }

void Model::reset() { impl_->cache.reset(); }

void Model::set_hidden_state_hook(HiddenStateHook hook) { impl_->hook = std::move(hook); }

}  // namespace minillm
