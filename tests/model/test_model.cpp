// Model — the assembled forward pass.
//
// Two halves:
//
//   * A tiny random model, built in memory, for the contracts that need no
//     checkpoint: every constructor and forward() error, the hook protocol,
//     the arena sizing, tied and untied LM heads, and a whole forward pass
//     (prefill and decode) against one assembled from minillm::ref.
//   * SmolLM2-135M against oracle/data/model/ and oracle/data/attention/:
//     every hidden state in layer order, the final logits, the 64-token
//     greedy generation id for id, prefill-then-decode against one-shot
//     prefill, reset(), and the post-RoPE keys the cache ends up holding.
//   * Q8_0 weights (quantize_weights_q8_0): which tensors it replaces and
//     what the constructor accepts, on the tiny model; on SmolLM2, the C++
//     quantizer's bytes against oracle/generate.py's for the whole
//     checkpoint, the same oracle comparisons against oracle/data/model_q8_0/
//     (torch on the rounded weights, so the same tolerances and the same
//     token-exact greedy ids as fp32), and, on the release preset only, the
//     format's quality budget: KL divergence from the fp32 engine.
//
// The SmolLM2 cases print the measured error per prompt (max abs, max rel,
// and the share of the tolerance the worst element uses). Run them on the
// release preset to see the real numbers: debug and asan reproduce torch's
// matmuls bit for bit, so they report errors far below what release gets.
//
// Under ThreadSanitizer the oracle comparisons run fewer prompts and tokens
// (see kTsan): TSan runs the forward pass about 40 times slower than release,
// and its job here is the threaded kernels, which a shorter run exercises
// just as well. asan and release run everything.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../../src/model/scratch.h"
#include "common/allclose.h"
#include "common/oracle.h"
#include "common/random.h"
#include "common/tolerances.h"
#include "minillm/config.h"
#include "minillm/kv_cache.h"
#include "minillm/loader.h"
#include "minillm/model.h"
#include "minillm/ops.h"
#include "minillm/runtime.h"
#include "q8_0_eval_ids.h"

// Sanitizers: GCC defines the macros, Clang has the feature tests.
#if defined(__SANITIZE_THREAD__)
#define MINILLM_TEST_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define MINILLM_TEST_TSAN 1
#endif
#endif
#ifndef MINILLM_TEST_TSAN
#define MINILLM_TEST_TSAN 0
#endif
#if defined(__SANITIZE_ADDRESS__)
#define MINILLM_TEST_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define MINILLM_TEST_ASAN 1
#endif
#endif
#ifndef MINILLM_TEST_ASAN
#define MINILLM_TEST_ASAN 0
#endif
#if defined(NDEBUG) && !MINILLM_TEST_TSAN && !MINILLM_TEST_ASAN
#define MINILLM_TEST_RELEASE 1
#else
#define MINILLM_TEST_RELEASE 0
#endif

using namespace minillm;
using namespace minillm::test;
namespace fs = std::filesystem;

namespace {

bool contains(std::string_view haystack, std::string_view needle) {
  return haystack.find(needle) != std::string_view::npos;
}

std::span<const float> span_of(const Tensor& t) {
  return {t.f32(), static_cast<std::size_t>(t.numel())};
}

std::int32_t argmax_of(std::span<const float> logits) {
  const Tensor row = Tensor::wrap(const_cast<float*>(logits.data()),
                                  Shape{static_cast<std::int64_t>(logits.size())});
  return ops::argmax(row);
}

/// The share of `tol` that the worst element uses: max |a - e| / (atol + rtol |e|).
/// Above 1 means the comparison fails.
double tolerance_use(std::span<const float> actual, std::span<const float> expected,
                     Tolerance tol) {
  double worst = 0.0;
  for (std::size_t i = 0; i < actual.size() && i < expected.size(); ++i) {
    const double e = expected[i];
    const double bound = tol.atol + tol.rtol * std::fabs(e);
    worst = std::max(worst, std::fabs(static_cast<double>(actual[i]) - e) / bound);
  }
  return worst;
}

/// The what() of the E that f() throws. Fails the enclosing test if f returns,
/// or throws anything else.
template <class E, class F>
std::string thrown(F&& f) {
  try {
    f();
  } catch (const E& e) {
    return e.what();
  } catch (const std::exception& e) {
    FAIL_CHECK("threw the wrong exception type: " << e.what());
    return {};
  }
  FAIL_CHECK("returned without throwing");
  return {};
}

/// SHA-256 (FIPS 180-4) of `size` bytes, as 64 lowercase hex digits: the
/// digest oracle/generate.py writes with hashlib for each Q8_0 tensor.
std::string sha256_hex(const void* data, std::size_t size) {
  static constexpr std::uint32_t k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
      0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
      0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
      0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
      0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
      0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
      0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
      0xc67178f2};
  std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  const auto rotr = [](std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
  const auto compress = [&](const unsigned char* p) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = static_cast<std::uint32_t>(p[4 * i]) << 24 |
             static_cast<std::uint32_t>(p[4 * i + 1]) << 16 |
             static_cast<std::uint32_t>(p[4 * i + 2]) << 8 |
             static_cast<std::uint32_t>(p[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t v[8];
    std::copy(h, h + 8, v);
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25);
      const std::uint32_t ch = (v[4] & v[5]) ^ (~v[4] & v[6]);
      const std::uint32_t t1 = v[7] + s1 + ch + k[i] + w[i];
      const std::uint32_t s0 = rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22);
      const std::uint32_t maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
      std::copy_backward(v, v + 7, v + 8);
      v[4] += t1;
      v[0] = t1 + s0 + maj;
    }
    for (int i = 0; i < 8; ++i) h[i] += v[i];
  };

  const auto* bytes = static_cast<const unsigned char*>(data);
  const std::size_t whole = size / 64 * 64;
  for (std::size_t i = 0; i < whole; i += 64) compress(bytes + i);
  // The rest, a 1 bit, zeros, and the length in bits as a big-endian uint64.
  unsigned char tail[128] = {};
  const std::size_t rest = size - whole;
  if (rest > 0) std::memcpy(tail, bytes + whole, rest);
  tail[rest] = 0x80;
  const std::size_t tail_size = rest + 1 + 8 <= 64 ? 64 : 128;
  const std::uint64_t bits = static_cast<std::uint64_t>(size) * 8;
  for (int i = 0; i < 8; ++i) tail[tail_size - 1 - i] = static_cast<unsigned char>(bits >> (8 * i));
  for (std::size_t i = 0; i < tail_size; i += 64) compress(tail + i);

  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  for (const std::uint32_t word : h) {
    for (int shift = 28; shift >= 0; shift -= 4) out += kHex[(word >> shift) & 0xf];
  }
  return out;
}

/// Copies every hidden state the model reports, with its index.
struct HookRecorder {
  std::vector<std::int64_t> indices;
  std::vector<Tensor> states;

  Model::HiddenStateHook hook() {
    return [this](std::int64_t index, const Tensor& hidden) {
      indices.push_back(index);
      states.push_back(hidden.copy_contiguous());
    };
  }
  void clear() {
    indices.clear();
    states.clear();
  }
};

// ---------------------------------------------------------------------------
// A tiny random model
// ---------------------------------------------------------------------------

/// Every dimension distinct, so a shape in an error message names one kind of
/// tensor; GQA (2 query heads per KV head) and head_dim * n_heads != hidden,
/// so neither can be assumed by accident.
ModelConfig tiny_config() {
  ModelConfig c;
  c.vocab_size = 40;
  c.hidden_size = 12;
  c.intermediate_size = 20;
  c.num_hidden_layers = 2;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 4;
  c.max_position_embeddings = 32;
  c.rms_norm_eps = 1e-5f;
  c.rope_theta = 10000.0f;
  c.tie_word_embeddings = true;
  c.model_type = "llama";
  return c;
}

/// Every tensor loader.h lists for `c`, with its shape, in loader.h order.
std::vector<std::pair<std::string, Shape>> tensor_shapes(const ModelConfig& c) {
  const std::int64_t h = c.hidden_size;
  const std::int64_t inter = c.intermediate_size;
  std::vector<std::pair<std::string, Shape>> out;
  out.emplace_back("model.embed_tokens.weight", Shape{c.vocab_size, h});
  for (std::int64_t i = 0; i < c.num_hidden_layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    out.emplace_back(p + "input_layernorm.weight", Shape{h});
    out.emplace_back(p + "self_attn.q_proj.weight", Shape{c.q_dim(), h});
    out.emplace_back(p + "self_attn.k_proj.weight", Shape{c.kv_dim(), h});
    out.emplace_back(p + "self_attn.v_proj.weight", Shape{c.kv_dim(), h});
    out.emplace_back(p + "self_attn.o_proj.weight", Shape{h, c.q_dim()});
    out.emplace_back(p + "post_attention_layernorm.weight", Shape{h});
    out.emplace_back(p + "mlp.gate_proj.weight", Shape{inter, h});
    out.emplace_back(p + "mlp.up_proj.weight", Shape{inter, h});
    out.emplace_back(p + "mlp.down_proj.weight", Shape{h, inter});
  }
  out.emplace_back("model.norm.weight", Shape{h});
  if (!c.tie_word_embeddings) out.emplace_back("lm_head.weight", Shape{c.vocab_size, h});
  return out;
}

/// Random weights for `c`: norm weights near 1 as in a real checkpoint, linear
/// weights scaled by 1/sqrt(fan_in) so activations stay O(1) through the layers.
Weights random_weights(const ModelConfig& c, std::uint64_t seed = 1) {
  Weights w;
  for (const auto& [name, shape] : tensor_shapes(c)) {
    ++seed;
    if (shape.ndim() == 1) {
      w.insert(name, random_tensor_uniform(shape, seed, 0.8f, 1.2f));
      continue;
    }
    Tensor t = random_tensor(shape, seed);
    if (name != "model.embed_tokens.weight") {
      const float scale = 1.0f / std::sqrt(static_cast<float>(shape[1]));
      for (std::int64_t i = 0; i < t.numel(); ++i) t.f32()[i] *= scale;
    }
    w.insert(name, std::move(t));
  }
  return w;
}

/// `w` minus `drop`, plus `add` (which replaces a tensor of the same name).
Weights edited(const Weights& w, std::initializer_list<std::string> drop,
               std::vector<std::pair<std::string, Tensor>> add = {}) {
  Weights out;
  for (const std::string& name : w.names()) {
    const bool dropped = std::find(drop.begin(), drop.end(), name) != drop.end();
    const bool replaced = std::any_of(add.begin(), add.end(),
                                      [&](const auto& kv) { return kv.first == name; });
    if (!dropped && !replaced) out.insert(name, w.get(name));
  }
  for (auto& [name, t] : add) out.insert(name, std::move(t));
  return out;
}

/// tiny_config() with every row a whole number of Q8_0 blocks, so that
/// quantize_weights_q8_0 accepts it: hidden 64, intermediate 96, head_dim 8
/// (q_dim 32, kv_dim 16). The rows are the last dimensions: hidden for
/// everything but o_proj (q_dim) and down_proj (intermediate).
ModelConfig tiny_q8_0_config() {
  ModelConfig c = tiny_config();
  c.hidden_size = 64;
  c.intermediate_size = 96;
  c.head_dim = 8;
  return c;
}

/// `w` with every Q8_0 tensor replaced by convert_to_f32 of it: the F32
/// weights a Q8_0 model stands for (ops.h).
Weights dequantized(const Weights& w) {
  Weights out;
  for (const std::string& name : w.names()) {
    const Tensor& t = w.get(name);
    out.insert(name, t.dtype() == DType::Q8_0 ? convert_to_f32(t) : t);
  }
  return out;
}

/// The what() of the std::runtime_error that constructing this model throws.
/// Fails the enclosing test if it constructs, or throws anything else.
std::string constructor_error(const ModelConfig& c, Weights w, std::int64_t max_seq_len = 16) {
  try {
    Model m(c, std::move(w), max_seq_len);
  } catch (const std::runtime_error& e) {
    return e.what();
  } catch (const std::exception& e) {
    FAIL_CHECK("expected std::runtime_error, got: " << e.what());
    return {};
  }
  FAIL_CHECK("the model constructed without throwing");
  return {};
}

/// The same for forward() and std::invalid_argument.
std::string forward_error(Model& m, std::span<const std::int32_t> tokens, std::int64_t start_pos) {
  try {
    m.forward(tokens, start_pos);
  } catch (const std::invalid_argument& e) {
    return e.what();
  } catch (const std::exception& e) {
    FAIL_CHECK("expected std::invalid_argument, got: " << e.what());
    return {};
  }
  FAIL_CHECK("forward() returned without throwing");
  return {};
}

/// A whole forward pass over `ids` from position 0, assembled from ref:: ops
/// with no KV cache: the obvious version of docs/ARCHITECTURE.md §4.
struct RefPass {
  std::vector<Tensor> hidden;  ///< num_hidden_layers + 1 states, [n, hidden]
  std::vector<float> logits;   ///< the last row, [vocab]
};

RefPass ref_forward(const ModelConfig& c, const Weights& w, std::span<const std::int32_t> ids) {
  const auto n = static_cast<std::int64_t>(ids.size());
  const std::int64_t hd = c.head_dim;
  std::vector<std::int64_t> positions(ids.size());
  std::iota(positions.begin(), positions.end(), std::int64_t{0});

  RefPass out;
  Tensor h = Tensor::zeros(Shape{n, c.hidden_size});
  ref::embedding(w.get("model.embed_tokens.weight"), ids, h);
  for (std::int64_t l = 0; l < c.num_hidden_layers; ++l) {
    out.hidden.push_back(h.copy_contiguous());
    const std::string p = "model.layers." + std::to_string(l) + ".";
    Tensor x = Tensor::zeros(Shape{n, c.hidden_size});
    ref::rmsnorm(h, w.get(p + "input_layernorm.weight"), c.rms_norm_eps, x);
    Tensor q = Tensor::zeros(Shape{n, c.num_attention_heads, hd});
    Tensor k = Tensor::zeros(Shape{n, c.num_key_value_heads, hd});
    Tensor v = Tensor::zeros(Shape{n, c.num_key_value_heads, hd});
    Tensor q2 = q.reshape(Shape{n, c.q_dim()});
    Tensor k2 = k.reshape(Shape{n, c.kv_dim()});
    Tensor v2 = v.reshape(Shape{n, c.kv_dim()});
    ref::matmul(x, w.get(p + "self_attn.q_proj.weight"), q2);
    ref::matmul(x, w.get(p + "self_attn.k_proj.weight"), k2);
    ref::matmul(x, w.get(p + "self_attn.v_proj.weight"), v2);
    ref::rope(q, &k, positions, c.rope_theta);
    Tensor a = Tensor::zeros(Shape{n, c.num_attention_heads, hd});
    ref::attention(q, k, v, 0, c.attention_scale(), a);
    Tensor o = Tensor::zeros(Shape{n, c.hidden_size});
    ref::matmul(a.reshape(Shape{n, c.q_dim()}), w.get(p + "self_attn.o_proj.weight"), o);
    ref::add(h, o, h);

    ref::rmsnorm(h, w.get(p + "post_attention_layernorm.weight"), c.rms_norm_eps, x);
    Tensor g = Tensor::zeros(Shape{n, c.intermediate_size});
    Tensor u = Tensor::zeros(Shape{n, c.intermediate_size});
    ref::matmul(x, w.get(p + "mlp.gate_proj.weight"), g);
    ref::matmul(x, w.get(p + "mlp.up_proj.weight"), u);
    ref::silu_mul(g, u, g);
    ref::matmul(g, w.get(p + "mlp.down_proj.weight"), o);
    ref::add(h, o, h);
  }
  ref::rmsnorm(h, w.get("model.norm.weight"), c.rms_norm_eps, h);
  out.hidden.push_back(h.copy_contiguous());

  const Tensor& lm_head =
      w.get(c.tie_word_embeddings ? "model.embed_tokens.weight" : "lm_head.weight");
  Tensor logits = Tensor::zeros(Shape{1, c.vocab_size});
  ref::matmul(h.slice(0, n - 1, 1), lm_head, logits);
  out.logits.assign(logits.f32(), logits.f32() + c.vocab_size);
  return out;
}

// ---------------------------------------------------------------------------
// SmolLM2-135M and its oracle data
// ---------------------------------------------------------------------------

const char* const kModel = "SmolLM2-135M";
constexpr int kPrompts = 3;
constexpr std::int64_t kGenerate = 64;
/// Room for the longest prompt (15 tokens) plus the whole greedy generation.
constexpr std::int64_t kMaxSeq = 128;

/// Records a skip, naming the files, if the checkpoint is not there.
bool skip_without_checkpoint() {
  std::string missing;
  for (const char* name : {"config.json", "model.safetensors"}) {
    const fs::path file = models_dir() / kModel / name;
    std::error_code ec;
    if (!fs::is_regular_file(file, ec)) {
      if (!missing.empty()) missing += ", ";
      missing += file.string();
    }
  }
  if (missing.empty()) return false;
  const std::string reason =
      "missing model file: " + missing + " — run `python oracle/generate.py` to download it";
  note_skip(reason);
  MESSAGE("SKIP: " << reason);
  return true;
}

const ModelConfig& smollm2_config() {
  static const ModelConfig cfg = ModelConfig::from_json_file(models_dir() / kModel / "config.json");
  return cfg;
}

/// Loaded once per process and shared by handle: every Model built from a
/// copy of this Weights points at the same storage.
const Weights& smollm2_weights() {
  static const Weights weights = load_checkpoint(models_dir() / kModel);
  return weights;
}

Model smollm2(std::int64_t max_seq_len = kMaxSeq) {
  return Model(smollm2_config(), smollm2_weights(), max_seq_len);
}

/// smollm2_weights() through quantize_weights_q8_0, once per process. The
/// norms still share storage with smollm2_weights(); replace() only swapped
/// this copy's handles to the projections and the embedding.
const Weights& smollm2_q8_0_weights() {
  static const Weights weights = [] {
    Weights w = smollm2_weights();
    quantize_weights_q8_0(w, smollm2_config());
    return w;
  }();
  return weights;
}

/// One weight format and the oracle section that holds its golden data.
struct Format {
  const char* name;     ///< "f32" or "q8_0", for messages
  const char* section;  ///< "model" or "model_q8_0" under oracle/data/
  const Weights& (*weights)();
};
const Format kF32{"f32", "model", smollm2_weights};
const Format kQ8_0{"q8_0", "model_q8_0", smollm2_q8_0_weights};

Model smollm2(const Format& format, std::int64_t max_seq_len = kMaxSeq) {
  return Model(smollm2_config(), format.weights(), max_seq_len);
}

std::string prompt_file(int p, const std::string& what, const std::string& section = "model") {
  return section + "/p" + std::to_string(p) + "_" + what + ".npy";
}

std::vector<std::string> prompt_files(int p, const std::string& section = "model") {
  std::vector<std::string> files = {prompt_file(p, "input_ids", section),
                                    prompt_file(p, "logits", section),
                                    prompt_file(p, "greedy_ids", section)};
  for (std::int64_t j = 0; j <= smollm2_config().num_hidden_layers; ++j) {
    files.push_back(prompt_file(p, "hidden_" + std::to_string(j), section));
  }
  return files;
}

std::vector<std::int32_t> prompt_ids(int p, const std::string& section = "model") {
  return oracle_npy(prompt_file(p, "input_ids", section)).as_i32_narrowing();
}

constexpr bool kTsan = MINILLM_TEST_TSAN;
constexpr bool kRelease = MINILLM_TEST_RELEASE;

/// Under TSan the oracle comparisons run only the longest prompt, p2 (15
/// tokens, the one whose prefill splits attention across threads), and its
/// greedy generation stops after kTsanTokens: from the 8th decode step on,
/// attention over more than 22 positions is split across threads too.
constexpr std::int64_t kTsanTokens = 16;

std::vector<int> prompts_to_check() {
  if (kTsan) return {2};
  return {0, 1, 2};
}

std::int64_t greedy_tokens_to_check() { return kTsan ? kTsanTokens : kGenerate; }

std::string hidden_state_name(std::int64_t j, std::int64_t n_layers) {
  if (j == 0) return "hidden state 0 (the embedding output)";
  if (j == n_layers) return "hidden state " + std::to_string(j) + " (after the final norm)";
  return "hidden state " + std::to_string(j) + " (the input to layer " + std::to_string(j) + ")";
}

}  // namespace

// ===========================================================================
// Arena sizing
// ===========================================================================

TEST_CASE("scratch_bytes is exactly what carve takes from the arena") {
  ModelConfig smollm2_shape = tiny_config();
  smollm2_shape.vocab_size = 49152;
  smollm2_shape.hidden_size = 576;
  smollm2_shape.intermediate_size = 1536;
  smollm2_shape.num_hidden_layers = 30;
  smollm2_shape.num_attention_heads = 9;
  smollm2_shape.num_key_value_heads = 3;
  smollm2_shape.head_dim = 64;
  smollm2_shape.max_position_embeddings = 8192;

  for (const ModelConfig& cfg : {tiny_config(), smollm2_shape}) {
    for (const std::int64_t n : {1, 5, 7, 192}) {
      CAPTURE(cfg.hidden_size);
      CAPTURE(n);
      Arena arena(model_detail::scratch_bytes(cfg, n));
      const model_detail::Scratch s = model_detail::carve(arena, cfg, n);
      CHECK(arena.peak() == arena.capacity());
      CHECK(s.h.shape() == Shape{n, cfg.hidden_size});
      CHECK(s.logits.shape() == Shape{1, cfg.vocab_size});
      // The constructor sizes for max_seq_len, so every shorter pass must fit.
      CHECK(model_detail::scratch_bytes(cfg, n) <= model_detail::scratch_bytes(cfg, n + 1));
    }
  }
}

TEST_CASE("a prefill of exactly max_seq_len tokens fits the arena and the cache") {
  const ModelConfig cfg = tiny_config();
  Model model(cfg, random_weights(cfg), 12);
  REQUIRE(model.kv_cache().max_seq_len() == 12);
  const std::vector<std::int32_t> ids = random_ids(12, 40, 7);
  CHECK_NOTHROW(model.forward(ids, 0));
  CHECK(model.kv_cache().seq_len() == 12);
  const std::int32_t one = 3;
  CHECK_THROWS_AS(model.forward(std::span(&one, 1), 12), std::invalid_argument);
}

// ===========================================================================
// Constructor
// ===========================================================================

TEST_CASE("a valid model reports its config and an empty cache sized for it") {
  const ModelConfig cfg = tiny_config();
  const Model model(cfg, random_weights(cfg), 16);
  CHECK(model.config().hidden_size == cfg.hidden_size);
  CHECK(model.config().num_hidden_layers == cfg.num_hidden_layers);
  const KVCache& cache = model.kv_cache();
  CHECK(cache.n_layers() == cfg.num_hidden_layers);
  CHECK(cache.max_seq_len() == 16);
  CHECK(cache.n_kv_heads() == cfg.num_key_value_heads);
  CHECK(cache.head_dim() == cfg.head_dim);
  CHECK(cache.seq_len() == 0);
}

TEST_CASE("max_seq_len is capped at max_position_embeddings, and must be positive") {
  const ModelConfig cfg = tiny_config();
  {
    Model model(cfg, random_weights(cfg), 1000);
    CHECK(model.kv_cache().max_seq_len() == cfg.max_position_embeddings);
    const std::vector<std::int32_t> ids = random_ids(32, 40, 3);
    CHECK_NOTHROW(model.forward(ids, 0));
  }
  CHECK_THROWS_AS(Model(cfg, random_weights(cfg), 0), std::invalid_argument);
  CHECK_THROWS_AS(Model(cfg, random_weights(cfg), -1), std::invalid_argument);
  // The argument is checked before any tensor: with no weights at all, the
  // error is still about max_seq_len (from_dir relies on the same check to
  // fail before reading the checkpoint).
  CHECK_THROWS_WITH_AS(Model(cfg, Weights{}, 0), doctest::Contains("max_seq_len must be positive"),
                       std::invalid_argument);
}

TEST_CASE("the config is validated before anything else") {
  ModelConfig cfg = tiny_config();
  cfg.hidden_act = "gelu";  // the tensors are all fine; only validate() can object
  CHECK(contains(constructor_error(cfg, random_weights(tiny_config())), "hidden_act"));
}

TEST_CASE("an odd head_dim is rejected at construction, not at the first RoPE") {
  ModelConfig cfg = tiny_config();
  cfg.head_dim = 3;  // q_dim 12, kv_dim 6: shapes consistent, but RoPE needs pairs
  CHECK(contains(constructor_error(cfg, random_weights(cfg)), "head_dim must be even"));
}

TEST_CASE("every missing tensor is reported by name with its expected shape") {
  for (const bool tied : {true, false}) {
    ModelConfig cfg = tiny_config();
    cfg.tie_word_embeddings = tied;
    const Weights w = random_weights(cfg);
    for (const auto& [name, shape] : tensor_shapes(cfg)) {
      CAPTURE(name);
      const std::string msg = constructor_error(cfg, edited(w, {name}));
      CHECK(contains(msg, "missing tensor '" + name + "'"));
      CHECK(contains(msg, "f32" + shape.to_string()));
    }
  }
}

TEST_CASE("every misshapen tensor is reported with its actual and expected shapes") {
  for (const bool tied : {true, false}) {
    ModelConfig cfg = tiny_config();
    cfg.tie_word_embeddings = tied;
    const Weights w = random_weights(cfg);
    for (const auto& [name, shape] : tensor_shapes(cfg)) {
      CAPTURE(name);
      Shape wrong = shape;
      wrong[0] += 1;
      const std::string msg = constructor_error(cfg, edited(w, {name}, {{name, Tensor::zeros(wrong)}}));
      CHECK(contains(msg, "'" + name + "'"));
      CHECK(contains(msg, "f32" + wrong.to_string()));
      CHECK(contains(msg, "expected f32" + shape.to_string()));
    }
  }
}

TEST_CASE("a tensor that is not f32, or not contiguous, is rejected by name") {
  const ModelConfig cfg = tiny_config();
  const Weights w = random_weights(cfg);
  const std::string name = "model.layers.1.self_attn.k_proj.weight";
  const Shape shape{cfg.kv_dim(), cfg.hidden_size};

  const std::string bf16 =
      constructor_error(cfg, edited(w, {name}, {{name, Tensor::zeros(shape, DType::BF16)}}));
  CHECK(contains(bf16, "'" + name + "' is bf16"));

  const Tensor strided = Tensor::zeros(Shape{cfg.kv_dim(), 2 * cfg.hidden_size})
                             .slice(1, 0, cfg.hidden_size);
  REQUIRE_FALSE(strided.is_contiguous());
  const std::string msg = constructor_error(cfg, edited(w, {name}, {{name, strided}}));
  CHECK(contains(msg, "'" + name + "' is not contiguous"));
}

TEST_CASE("a tensor the config does not describe is an error, not ignored") {
  const ModelConfig cfg = tiny_config();
  const Weights w = random_weights(cfg);
  // The likeliest way to get one: a checkpoint with more layers than the config.
  const std::string extra = "model.layers.2.input_layernorm.weight";
  const std::string msg = constructor_error(
      cfg, edited(w, {}, {{extra, random_tensor(Shape{cfg.hidden_size}, 99)}}));
  CHECK(contains(msg, "unexpected tensor '" + extra + "'"));

  const std::string derived = "model.layers.0.self_attn.rotary_emb.inv_freq";
  CHECK(contains(constructor_error(cfg, edited(w, {}, {{derived, Tensor::zeros(Shape{2})}})),
                 "unexpected tensor '" + derived + "'"));
}

TEST_CASE("tied embeddings: the LM head is embed_tokens, and a stray lm_head.weight must equal it") {
  const ModelConfig cfg = tiny_config();
  REQUIRE(cfg.tie_word_embeddings);
  const Weights w = random_weights(cfg);
  const std::vector<std::int32_t> ids = random_ids(6, 40, 11);

  Model plain(cfg, w, 16);
  const std::span<const float> l = plain.forward(ids, 0);
  const std::vector<float> expected(l.begin(), l.end());

  SUBCASE("an identical copy, in its own storage, is accepted and changes nothing") {
    Tensor copy = w.get("model.embed_tokens.weight").copy_contiguous();
    Model model(cfg, edited(w, {}, {{"lm_head.weight", copy}}), 16);
    const std::span<const float> got = model.forward(ids, 0);
    CHECK(std::equal(got.begin(), got.end(), expected.begin(), expected.end()));
  }
  SUBCASE("a copy that differs in one element is rejected") {
    Tensor copy = w.get("model.embed_tokens.weight").copy_contiguous();
    copy.f32()[123] += 1e-3f;
    const std::string msg = constructor_error(cfg, edited(w, {}, {{"lm_head.weight", copy}}));
    CHECK(contains(msg, "lm_head.weight differs from model.embed_tokens.weight"));
  }
  SUBCASE("a misshapen one is rejected") {
    const Tensor wrong = Tensor::zeros(Shape{cfg.vocab_size + 1, cfg.hidden_size});
    const std::string msg = constructor_error(cfg, edited(w, {}, {{"lm_head.weight", wrong}}));
    CHECK(contains(msg, "'lm_head.weight' is f32[41, 12]"));
  }
}

// ===========================================================================
// forward(): arguments
// ===========================================================================

TEST_CASE("forward() rejects bad arguments and leaves the cache and the last logits alone") {
  const ModelConfig cfg = tiny_config();
  Model model(cfg, random_weights(cfg), 8);
  const std::vector<std::int32_t> ids = random_ids(5, 40, 5);
  const std::span<const float> logits = model.forward(ids, 0);
  const std::vector<float> before(logits.begin(), logits.end());

  auto unchanged = [&] {
    CHECK(model.kv_cache().seq_len() == 5);
    // Reading the old span is itself a check: under ASan, rewinding the arena
    // poisons it, so a forward() that reset scratch before failing is caught.
    CHECK(std::equal(logits.begin(), logits.end(), before.begin(), before.end()));
  };

  SUBCASE("no tokens") {
    CHECK(contains(forward_error(model, {}, 5), "tokens is empty"));
    unchanged();
  }
  SUBCASE("an id below 0 or at vocab_size") {
    for (const std::int32_t bad : {-1, 40}) {
      const std::vector<std::int32_t> t = {1, bad};
      CHECK(contains(forward_error(model, t, 5), "token id " + std::to_string(bad) + " at index 1"));
    }
    unchanged();
  }
  SUBCASE("a negative start_pos") {
    const std::int32_t one = 1;
    CHECK(contains(forward_error(model, std::span(&one, 1), -1), "negative"));
    unchanged();
  }
  SUBCASE("a start_pos past the cached positions") {
    const std::int32_t one = 1;
    CHECK(contains(forward_error(model, std::span(&one, 1), 6), "past the 5 cached positions"));
    unchanged();
    CHECK_NOTHROW(model.forward(std::span(&one, 1), 5));
  }
  SUBCASE("start_pos + n beyond the cache") {
    const std::vector<std::int32_t> four = {1, 2, 3, 4};
    CHECK(contains(forward_error(model, four, 5), "exceeds the KV cache's 8 positions"));
    unchanged();
    CHECK_NOTHROW(model.forward(std::span(four).first(3), 5));
    CHECK(model.kv_cache().seq_len() == 8);
  }
}

TEST_CASE("a start_pos before the cache length re-runs from there") {
  const ModelConfig cfg = tiny_config();
  const Weights w = random_weights(cfg);
  const std::vector<std::int32_t> a = random_ids(7, 40, 21);
  const std::vector<std::int32_t> b = random_ids(4, 40, 22);

  Model fresh(cfg, w, 16);
  const std::span<const float> l = fresh.forward(b, 0);
  const std::vector<float> expected(l.begin(), l.end());

  Model reused(cfg, w, 16);
  reused.forward(a, 0);
  const std::span<const float> got = reused.forward(b, 0);
  CHECK(reused.kv_cache().seq_len() == 4);
  CHECK(std::equal(got.begin(), got.end(), expected.begin(), expected.end()));
}

// ===========================================================================
// forward(): the hook and the numbers, on the tiny model
// ===========================================================================

TEST_CASE("the hook is called num_hidden_layers + 1 times, in order, with [n, hidden] views") {
  const ModelConfig cfg = tiny_config();
  const Weights w = random_weights(cfg);
  Model model(cfg, w, 16);
  HookRecorder rec;
  model.set_hidden_state_hook(rec.hook());
  const std::vector<std::int32_t> ids = random_ids(5, 40, 13);

  model.forward(ids, 0);
  REQUIRE(rec.indices.size() == static_cast<std::size_t>(cfg.num_hidden_layers + 1));
  for (std::size_t j = 0; j < rec.indices.size(); ++j) {
    CHECK(rec.indices[j] == static_cast<std::int64_t>(j));
    CHECK(rec.states[j].shape() == Shape{5, cfg.hidden_size});
  }
  // Index 0 is the embedding output, exactly.
  const Tensor& embed = w.get("model.embed_tokens.weight");
  for (std::int64_t t = 0; t < 5; ++t) {
    const float* row = embed.f32() + ids[static_cast<std::size_t>(t)] * cfg.hidden_size;
    CHECK(std::equal(row, row + cfg.hidden_size, rec.states[0].f32() + t * cfg.hidden_size));
  }

  rec.clear();
  const std::int32_t next = 9;
  model.forward(std::span(&next, 1), 5);
  REQUIRE(rec.indices.size() == static_cast<std::size_t>(cfg.num_hidden_layers + 1));
  CHECK(rec.states.back().shape() == Shape{1, cfg.hidden_size});

  rec.clear();
  model.set_hidden_state_hook({});
  model.forward(std::span(&next, 1), 6);
  CHECK(rec.indices.empty());
}

TEST_CASE("the tiny model matches a ref:: forward pass, tied and untied, prefill and decode") {
  for (const bool tied : {true, false}) {
    CAPTURE(tied);
    ModelConfig cfg = tiny_config();
    cfg.tie_word_embeddings = tied;
    const Weights w = random_weights(cfg, tied ? 100 : 200);
    const std::vector<std::int32_t> ids = random_ids(9, 40, 17);
    const RefPass expected = ref_forward(cfg, w, ids);

    SUBCASE("one-shot prefill: every hidden state and the logits") {
      Model model(cfg, w, 16);
      HookRecorder rec;
      model.set_hidden_state_hook(rec.hook());
      const std::span<const float> logits = model.forward(ids, 0);
      REQUIRE(rec.states.size() == expected.hidden.size());
      for (std::size_t j = 0; j < rec.states.size(); ++j) {
        assert_allclose(rec.states[j], expected.hidden[j], tol::kHiddenState,
                        "tiny " + hidden_state_name(static_cast<std::int64_t>(j), 2));
      }
      assert_allclose(logits, expected.logits, tol::kLogits, "tiny logits");
    }
    SUBCASE("prefill of 6, then three one-token decode steps") {
      Model model(cfg, w, 16);
      HookRecorder rec;
      model.set_hidden_state_hook(rec.hook());
      model.forward(std::span(ids).first(6), 0);
      std::span<const float> logits;
      for (std::int64_t t = 6; t < 9; ++t) {
        rec.clear();
        logits = model.forward(std::span(ids).subspan(static_cast<std::size_t>(t), 1), t);
        REQUIRE(rec.states.size() == expected.hidden.size());
        for (std::size_t j = 0; j < rec.states.size(); ++j) {
          assert_allclose(rec.states[j], expected.hidden[j].slice(0, t, 1).copy_contiguous(),
                          tol::kHiddenState,
                          "tiny decode at " + std::to_string(t) + ", " +
                              hidden_state_name(static_cast<std::int64_t>(j), 2));
        }
      }
      assert_allclose(logits, expected.logits, tol::kLogits, "tiny logits after decode");
    }
  }
}

// ===========================================================================
// Q8_0 weights, on the tiny model
// ===========================================================================

TEST_CASE("quantize_weights_q8_0 replaces the projections and the embedding, and nothing else") {
  for (const bool tied : {true, false}) {
    CAPTURE(tied);
    ModelConfig cfg = tiny_q8_0_config();
    cfg.tie_word_embeddings = tied;
    const Weights f32 = random_weights(cfg);
    Weights w = f32;
    quantize_weights_q8_0(w, cfg);

    REQUIRE(w.names() == f32.names());
    std::size_t bytes = 0;
    for (const auto& [name, shape] : tensor_shapes(cfg)) {
      CAPTURE(name);
      const Tensor& t = w.get(name);
      CHECK(t.shape() == shape);
      bytes += t.nbytes();
      if (shape.ndim() == 1) {
        // A norm weight: the very same tensor, not a copy.
        CHECK(t.dtype() == DType::F32);
        CHECK(t.data() == f32.get(name).data());
        continue;
      }
      REQUIRE(t.dtype() == DType::Q8_0);
      const Tensor expected = quantize_q8_0(f32.get(name));
      CHECK(std::memcmp(t.data(), expected.data(), expected.nbytes()) == 0);
    }
    CHECK(w.bytes() == bytes);
    // replace() swapped this copy's handles; the original still holds F32.
    CHECK(f32.get("model.embed_tokens.weight").dtype() == DType::F32);
  }
}

TEST_CASE("quantize_weights_q8_0 names a missing tensor before it replaces anything") {
  const ModelConfig cfg = tiny_q8_0_config();
  const std::string gone = "model.layers.1.mlp.down_proj.weight";
  Weights w = edited(random_weights(cfg), {gone});
  CHECK(contains(thrown<std::runtime_error>([&] { quantize_weights_q8_0(w, cfg); }), gone));
  for (const std::string& name : w.names()) {
    CAPTURE(name);
    CHECK(w.get(name).dtype() == DType::F32);
  }

  // Untied, lm_head.weight is one of the tensors it needs.
  ModelConfig untied = cfg;
  untied.tie_word_embeddings = false;
  Weights u = edited(random_weights(untied), {"lm_head.weight"});
  CHECK(contains(thrown<std::runtime_error>([&] { quantize_weights_q8_0(u, untied); }),
                 "lm_head.weight"));
}

TEST_CASE("quantize_weights_q8_0 names the tensor quantize_q8_0 rejects") {
  SUBCASE("rows that are not a whole number of blocks") {
    const ModelConfig cfg = tiny_config();  // hidden 12
    Weights w = random_weights(cfg);
    const std::string msg =
        thrown<std::invalid_argument>([&] { quantize_weights_q8_0(w, cfg); });
    CHECK(contains(msg, "quantize_weights_q8_0: tensor 'model.embed_tokens.weight'"));
    CHECK(contains(msg, "multiple of 32"));
  }
  SUBCASE("a non-finite weight") {
    const ModelConfig cfg = tiny_q8_0_config();
    const std::string name = "model.layers.0.self_attn.v_proj.weight";
    const Weights good = random_weights(cfg);
    Tensor bad = good.get(name).copy_contiguous();
    bad.f32()[5] = std::numeric_limits<float>::infinity();
    Weights w = edited(good, {}, {{name, bad}});
    const std::string msg =
        thrown<std::invalid_argument>([&] { quantize_weights_q8_0(w, cfg); });
    CHECK(contains(msg, "tensor '" + name + "'"));
    CHECK(contains(msg, "finite"));
  }
}

TEST_CASE("the constructor takes each projection and the embedding as f32 or q8_0, norms as f32") {
  for (const bool tied : {true, false}) {
    CAPTURE(tied);
    ModelConfig cfg = tiny_q8_0_config();
    cfg.tie_word_embeddings = tied;
    const Weights w = random_weights(cfg);
    Weights all = w;
    quantize_weights_q8_0(all, cfg);
    CHECK_NOTHROW(Model(cfg, all, 16));

    // One tensor at a time, the rest F32: every mix is a valid model.
    for (const auto& [name, shape] : tensor_shapes(cfg)) {
      CAPTURE(name);
      Weights one = edited(w, {}, {{name, quantize_q8_0(w.get(name))}});
      if (shape.ndim() > 1) {
        CHECK_NOTHROW(Model(cfg, std::move(one), 16));
      } else {
        CHECK(contains(constructor_error(cfg, std::move(one)),
                       "'" + name + "' is q8_0" + shape.to_string() + ", expected f32" +
                           shape.to_string()));
      }
    }
  }
}

TEST_CASE("a projection of the wrong dtype or shape is reported against what the slot takes") {
  const ModelConfig cfg = tiny_q8_0_config();
  const Weights w = random_weights(cfg);
  const std::string name = "model.layers.1.mlp.up_proj.weight";
  const Shape shape{cfg.intermediate_size, cfg.hidden_size};
  CHECK(contains(constructor_error(cfg, edited(w, {}, {{name, Tensor::zeros(shape, DType::BF16)}})),
                 "'" + name + "' is bf16[96, 64], expected f32 or q8_0[96, 64]"));
  const Shape wrong{cfg.intermediate_size + 1, cfg.hidden_size};
  CHECK(contains(constructor_error(cfg, edited(w, {}, {{name, Tensor::zeros(wrong, DType::Q8_0)}})),
                 "'" + name + "' is q8_0[97, 64], expected q8_0[96, 64]"));
}

TEST_CASE("tied Q8_0 embeddings: a stray lm_head.weight is quantized too, and must match it") {
  const ModelConfig cfg = tiny_q8_0_config();
  REQUIRE(cfg.tie_word_embeddings);
  const Weights w = random_weights(cfg);
  const Tensor copy = w.get("model.embed_tokens.weight").copy_contiguous();

  Weights both = edited(w, {}, {{"lm_head.weight", copy}});
  quantize_weights_q8_0(both, cfg);
  CHECK(both.get("lm_head.weight").dtype() == DType::Q8_0);
  CHECK_NOTHROW(Model(cfg, both, 16));

  Weights quantized = w;
  quantize_weights_q8_0(quantized, cfg);
  CHECK(contains(constructor_error(cfg, edited(quantized, {}, {{"lm_head.weight", copy}})),
                 "lm_head.weight is f32 and model.embed_tokens.weight q8_0"));
}

TEST_CASE("from_dir takes f32 or q8_0, and rejects any other dtype before reading anything") {
  // The directory does not exist: the dtype is checked first.
  for (const DType dt : {DType::BF16, DType::F16}) {
    CAPTURE(dtype_name(dt));
    const std::string msg = thrown<std::invalid_argument>(
        [&] { Model::from_dir("/nonexistent/minillm-model", 16, dt); });
    CHECK(contains(msg, std::string("weight_dtype must be f32 or q8_0, got ") + dtype_name(dt)));
  }
  CHECK_THROWS_AS(Model::from_dir("/nonexistent/minillm-model", 16, DType::Q8_0),
                  std::runtime_error);
}

TEST_CASE("the tiny Q8_0 model matches a ref:: forward pass on its dequantized weights") {
  for (const bool tied : {true, false}) {
    CAPTURE(tied);
    ModelConfig cfg = tiny_q8_0_config();
    cfg.tie_word_embeddings = tied;
    Weights w = random_weights(cfg, tied ? 300 : 400);
    quantize_weights_q8_0(w, cfg);
    const std::vector<std::int32_t> ids = random_ids(9, 40, 19);
    const RefPass expected = ref_forward(cfg, dequantized(w), ids);

    // One-shot prefill: every hidden state and the logits.
    Model model(cfg, w, 16);
    HookRecorder rec;
    model.set_hidden_state_hook(rec.hook());
    const std::span<const float> logits = model.forward(ids, 0);
    REQUIRE(rec.states.size() == expected.hidden.size());
    for (std::size_t j = 0; j < rec.states.size(); ++j) {
      assert_allclose(rec.states[j], expected.hidden[j], tol::kHiddenState,
                      "tiny q8_0 " + hidden_state_name(static_cast<std::int64_t>(j), 2));
    }
    assert_allclose(logits, expected.logits, tol::kLogits, "tiny q8_0 logits");

    // Prefill of 6, then three one-token decode steps: the m = 1 kernels.
    Model stepped(cfg, w, 16);
    stepped.forward(std::span(ids).first(6), 0);
    std::span<const float> last;
    for (std::int64_t t = 6; t < 9; ++t) {
      last = stepped.forward(std::span(ids).subspan(static_cast<std::size_t>(t), 1), t);
    }
    assert_allclose(last, expected.logits, tol::kLogits, "tiny q8_0 logits after decode");
  }
}

// ===========================================================================
// SmolLM2-135M against the oracle
// ===========================================================================

namespace {

/// Every hidden state of each prompt in prompts_to_check(), then the last
/// row's logits, against `format`'s oracle section; prints the measured
/// errors per prompt.
void check_hidden_states_and_logits(const Format& format) {
  const std::int64_t n_layers = smollm2_config().num_hidden_layers;
  const std::int64_t vocab = smollm2_config().vocab_size;
  const std::string section = format.section;

  for (const int p : prompts_to_check()) {
    MINILLM_SKIP_WITHOUT_ORACLE((prompt_files(p, section)));
    const std::vector<std::int32_t> ids = prompt_ids(p, section);
    const auto n = static_cast<std::int64_t>(ids.size());
    const std::string label = std::string(format.name) + " p" + std::to_string(p) + " (" +
                              std::to_string(n) + " tokens)";

    Model model = smollm2(format);
    HookRecorder rec;
    model.set_hidden_state_hook(rec.hook());
    const std::span<const float> logits = model.forward(ids, 0);
    REQUIRE(rec.states.size() == static_cast<std::size_t>(n_layers + 1));

    // In index order, stopping at the first failure: drift compounds, so the
    // first layer that is off is the one to debug and the rest is noise.
    double max_abs = 0.0, max_rel = 0.0, max_use = 0.0;
    std::int64_t abs_at = 0, use_at = 0;
    for (std::int64_t j = 0; j <= n_layers; ++j) {
      const Tensor& actual = rec.states[static_cast<std::size_t>(j)];
      const Tensor expected =
          oracle_npy(prompt_file(p, "hidden_" + std::to_string(j), section)).to_tensor();
      const CloseResult r = compare(actual, expected, tol::kHiddenState);
      const double use = tolerance_use(span_of(actual), span_of(expected), tol::kHiddenState);
      if (r.max_abs_error > max_abs) {
        max_abs = r.max_abs_error;
        abs_at = j;
      }
      if (use > max_use) {
        max_use = use;
        use_at = j;
      }
      max_rel = std::max(max_rel, r.max_rel_error);
      if (!r.ok) {
        assert_allclose(actual, expected, tol::kHiddenState,
                        label + ": " + hidden_state_name(j, n_layers) +
                            " is the first that drifts past tol::kHiddenState");
        break;
      }
    }

    const std::vector<float> all = oracle_npy(prompt_file(p, "logits", section)).as_f32();
    REQUIRE(all.size() == static_cast<std::size_t>(n * vocab));
    const std::span<const float> last(all.data() + (n - 1) * vocab, static_cast<std::size_t>(vocab));
    const CloseResult lr = compare(logits, last, tol::kLogits);
    assert_allclose(logits, last, tol::kLogits, label + ": last-row logits");

    MESSAGE(label << ": hidden states max abs " << max_abs << " (state " << abs_at
                  << "), max rel " << max_rel << ", worst " << 100.0 * max_use
                  << "% of kHiddenState (state " << use_at << "); logits max abs "
                  << lr.max_abs_error << ", max rel " << lr.max_rel_error << ", "
                  << 100.0 * tolerance_use(logits, last, tol::kLogits) << "% of kLogits");
  }
}

/// Greedy generation for each prompt in prompts_to_check(), the first
/// greedy_tokens_to_check() ids of `format`'s oracle, id for id.
void check_greedy_ids(const Format& format) {
  const std::string section = format.section;
  const std::int64_t n_tokens = greedy_tokens_to_check();
  Model model = smollm2(format);
  for (const int p : prompts_to_check()) {
    MINILLM_SKIP_WITHOUT_ORACLE({prompt_file(p, "input_ids", section),
                                 prompt_file(p, "greedy_ids", section)});
    const std::vector<std::int32_t> ids = prompt_ids(p, section);
    std::vector<std::int32_t> expected =
        oracle_npy(prompt_file(p, "greedy_ids", section)).as_i32_narrowing();
    REQUIRE(static_cast<std::int64_t>(expected.size()) == kGenerate);
    expected.resize(static_cast<std::size_t>(n_tokens));

    model.reset();
    std::span<const float> logits = model.forward(ids, 0);
    std::vector<std::int32_t> got;
    for (std::int64_t i = 0; i < n_tokens; ++i) {
      const std::int32_t next = argmax_of(logits);
      got.push_back(next);
      if (i + 1 < n_tokens) logits = model.forward(std::span(&next, 1), model.kv_cache().seq_len());
    }
    assert_equal(got, expected,
                 std::string(format.name) + " p" + std::to_string(p) + " greedy ids");
  }
}

/// For each prompt in prompts_to_check(): its prefill then one decode step
/// (the m = 1 kernels) against a one-shot prefill of the same tokens.
void check_decode_against_prefill(const Format& format) {
  for (const int p : prompts_to_check()) {
    MINILLM_SKIP_WITHOUT_ORACLE({prompt_file(p, "input_ids")});
    const std::vector<std::int32_t> ids = prompt_ids(p);
    const auto n = static_cast<std::int64_t>(ids.size());

    Model stepped = smollm2(format);
    const std::int32_t next = argmax_of(stepped.forward(ids, 0));
    const std::span<const float> d = stepped.forward(std::span(&next, 1), n);
    const std::vector<float> decoded(d.begin(), d.end());

    std::vector<std::int32_t> longer = ids;
    longer.push_back(next);
    Model oneshot = smollm2(format);
    const std::span<const float> once = oneshot.forward(longer, 0);

    const std::string label =
        std::string(format.name) + " p" + std::to_string(p) + " decode vs one-shot";
    CHECK_MESSAGE(argmax_of(decoded) == argmax_of(once), label);
    assert_allclose(decoded, once, tol::kLogits, label);
    CHECK(stepped.kv_cache().seq_len() == oneshot.kv_cache().seq_len());
  }
}

}  // namespace

TEST_CASE("SmolLM2: every hidden state matches torch in layer order, and so do the logits") {
  if (skip_without_checkpoint()) return;
  check_hidden_states_and_logits(kF32);
}

TEST_CASE("SmolLM2: 64 greedy tokens reproduce torch's, id for id, for every prompt") {
  if (skip_without_checkpoint()) return;
  check_greedy_ids(kF32);
}

TEST_CASE("SmolLM2: prefill then one decode step agrees with a one-shot prefill") {
  if (skip_without_checkpoint()) return;
  check_decode_against_prefill(kF32);
}

TEST_CASE("SmolLM2: after reset() a model runs a new prompt exactly as a fresh one does") {
  if (skip_without_checkpoint()) return;
  MINILLM_SKIP_WITHOUT_ORACLE({prompt_file(0, "input_ids"), prompt_file(1, "input_ids")});
  const std::vector<std::int32_t> first = prompt_ids(0);
  const std::vector<std::int32_t> second = prompt_ids(1);

  auto greedy_run = [&](Model& m) {
    std::span<const float> logits = m.forward(second, 0);
    std::vector<std::int32_t> out;
    for (int i = 0; i < 8; ++i) {
      out.push_back(argmax_of(logits));
      logits = m.forward(std::span(&out.back(), 1), m.kv_cache().seq_len());
    }
    return std::make_pair(out, std::vector<float>(logits.begin(), logits.end()));
  };

  Model fresh = smollm2();
  const auto expected = greedy_run(fresh);

  Model reused = smollm2();
  std::span<const float> l = reused.forward(first, 0);
  const std::int32_t t = argmax_of(l);
  reused.forward(std::span(&t, 1), reused.kv_cache().seq_len());
  reused.reset();
  CHECK(reused.kv_cache().seq_len() == 0);
  const auto got = greedy_run(reused);

  CHECK(got.first == expected.first);
  CHECK(got.second == expected.second);  // bit for bit: the same code on the same inputs
}

TEST_CASE("SmolLM2: the cache holds torch's post-RoPE keys and values, prefilled or decoded") {
  if (skip_without_checkpoint()) return;
  MINILLM_SKIP_WITHOUT_ORACLE({"attention/input_ids.npy", "attention/l0_k_rope.npy",
                               "attention/l0_v_proj.npy", "attention/l15_k_rope.npy",
                               "attention/l15_v_proj.npy"});
  const std::vector<std::int32_t> ids = oracle_npy("attention/input_ids.npy").as_i32_narrowing();
  const auto n = static_cast<std::int64_t>(ids.size());
  Model model = smollm2();

  SUBCASE("one prefill") { model.forward(ids, 0); }
  SUBCASE("a prefill, then the last token as a decode step") {
    model.forward(std::span(ids).first(ids.size() - 1), 0);
    model.forward(std::span(ids).last(1), n - 1);
  }
  REQUIRE(model.kv_cache().seq_len() == n);

  for (const int layer : {0, 15}) {
    const std::string tag = "attention/l" + std::to_string(layer);
    const std::vector<float> k = oracle_npy(tag + "_k_rope.npy").as_f32();
    const std::vector<float> v = oracle_npy(tag + "_v_proj.npy").as_f32();
    const Tensor ck = model.kv_cache().k(layer, n).copy_contiguous();
    const Tensor cv = model.kv_cache().v(layer, n).copy_contiguous();
    // Layer 15's input already carries 15 layers of drift.
    const Tolerance tol = layer == 0 ? tol::kProjection : tol::kHiddenState;
    assert_allclose(span_of(ck), k, tol, "layer " + std::to_string(layer) + " cached keys");
    assert_allclose(span_of(cv), v, tol, "layer " + std::to_string(layer) + " cached values");
  }
}

TEST_CASE("SmolLM2: from_dir loads the same model, and checks max_seq_len first") {
  if (skip_without_checkpoint()) return;
  MINILLM_SKIP_WITHOUT_ORACLE({prompt_file(0, "input_ids")});
  const std::vector<std::int32_t> ids = prompt_ids(0);

  Model loaded = Model::from_dir(models_dir() / kModel, 64);
  CHECK(loaded.config().num_hidden_layers == smollm2_config().num_hidden_layers);
  CHECK(loaded.kv_cache().max_seq_len() == 64);
  const std::span<const float> a = loaded.forward(ids, 0);
  const std::vector<float> from_dir(a.begin(), a.end());
  Model shared = smollm2(64);
  const std::span<const float> b = shared.forward(ids, 0);
  CHECK(std::equal(b.begin(), b.end(), from_dir.begin(), from_dir.end()));

  CHECK_THROWS_AS(Model::from_dir(models_dir() / kModel, 0), std::invalid_argument);
  CHECK_THROWS_AS(Model::from_dir(models_dir() / "no-such-model", 64), std::runtime_error);
}

TEST_CASE("SmolLM2: a moved-to model keeps its weights, cache and position") {
  if (skip_without_checkpoint()) return;
  MINILLM_SKIP_WITHOUT_ORACLE({prompt_file(2, "input_ids")});
  const std::vector<std::int32_t> ids = prompt_ids(2);
  const auto n = static_cast<std::int64_t>(ids.size());

  Model reference = smollm2();
  reference.forward(std::span(ids).first(ids.size() - 1), 0);
  const std::span<const float> r = reference.forward(std::span(ids).last(1), n - 1);
  const std::vector<float> expected(r.begin(), r.end());

  Model original = smollm2();
  original.forward(std::span(ids).first(ids.size() - 1), 0);
  Model moved = std::move(original);
  CHECK(moved.kv_cache().seq_len() == n - 1);
  const std::span<const float> got = moved.forward(std::span(ids).last(1), n - 1);
  CHECK(std::equal(got.begin(), got.end(), expected.begin(), expected.end()));
}

// ===========================================================================
// SmolLM2-135M with Q8_0 weights
// ===========================================================================

TEST_CASE("sha256_hex matches FIPS 180-4's examples") {
  CHECK(sha256_hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(sha256_hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  // 56 bytes: the length no longer fits the first padding block.
  const std::string two = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  CHECK(sha256_hex(two.data(), two.size()) ==
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  const std::string million(1000000, 'a');
  CHECK(sha256_hex(million.data(), million.size()) ==
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

// Not under TSan: quantizing and hashing are single-threaded, and the other
// presets already run them on every byte.
TEST_CASE("SmolLM2 Q8_0: the C++ quantizer writes the oracle's blocks for the whole checkpoint" *
          doctest::skip(kTsan)) {
  if (skip_without_checkpoint()) return;
  MINILLM_SKIP_WITHOUT_ORACLE({"model_q8_0/blocks.sha256"});

  // sha256sum's format: 64 hex digits, two spaces, the tensor name.
  std::map<std::string, std::string> expected;
  std::ifstream in(oracle_path("model_q8_0/blocks.sha256"));
  for (std::string line; std::getline(in, line);) {
    REQUIRE_MESSAGE(line.size() > 66, "malformed line in blocks.sha256: " << line);
    expected[line.substr(66)] = line.substr(0, 64);
  }

  const Weights& q = smollm2_q8_0_weights();
  std::size_t tensors = 0, blocks = 0, differ = 0;
  for (const std::string& name : q.names()) {
    const Tensor& t = q.get(name);
    if (t.dtype() != DType::Q8_0) continue;
    ++tensors;
    blocks += t.nbytes() / sizeof(BlockQ8_0);
    const auto it = expected.find(name);
    if (it == expected.end()) {
      FAIL_CHECK("C++ quantized '" << name << "', which blocks.sha256 does not list");
      continue;
    }
    if (sha256_hex(t.data(), t.nbytes()) != it->second) {
      ++differ;
      FAIL_CHECK("'" << name << "': its Q8_0 blocks differ from oracle/generate.py's");
    }
  }
  CHECK(tensors == expected.size());
  CHECK(differ == 0);
  MESSAGE("q8_0 blocks: " << tensors << " tensors, " << blocks << " blocks, " << differ
                          << " tensor(s) differ from the oracle's quantizer");
}

TEST_CASE("SmolLM2 Q8_0: every hidden state matches torch on Q8_0 weights, and so do the logits") {
  if (skip_without_checkpoint()) return;
  check_hidden_states_and_logits(kQ8_0);
}

TEST_CASE("SmolLM2 Q8_0: 64 greedy tokens reproduce torch's on the Q8_0 weights, id for id") {
  if (skip_without_checkpoint()) return;
  check_greedy_ids(kQ8_0);
}

TEST_CASE("SmolLM2 Q8_0: prefill then one decode step agrees with a one-shot prefill") {
  if (skip_without_checkpoint()) return;
  check_decode_against_prefill(kQ8_0);
}

TEST_CASE("SmolLM2 Q8_0: from_dir(..., Q8_0) loads the same model as quantize_weights_q8_0") {
  if (skip_without_checkpoint()) return;
  MINILLM_SKIP_WITHOUT_ORACLE({prompt_file(0, "input_ids")});
  const std::vector<std::int32_t> ids = prompt_ids(0);

  Model loaded = Model::from_dir(models_dir() / kModel, 64, DType::Q8_0);
  const std::span<const float> a = loaded.forward(ids, 0);
  const std::vector<float> from_dir(a.begin(), a.end());
  Model shared = smollm2(kQ8_0, 64);
  const std::span<const float> b = shared.forward(ids, 0);
  CHECK(std::equal(b.begin(), b.end(), from_dir.begin(), from_dir.end()));
}

// ---------------------------------------------------------------------------
// The format's quality budget (proposal 14, §5.3 layer 4)
// ---------------------------------------------------------------------------

namespace {

/// KL(p || q) in nats between the distributions two logit rows stand for,
/// in double.
double kl_divergence(std::span<const float> p_logits, std::span<const float> q_logits) {
  const auto log_sum_exp = [](std::span<const float> l) {
    const double m = *std::max_element(l.begin(), l.end());
    double sum = 0.0;
    for (const float x : l) sum += std::exp(static_cast<double>(x) - m);
    return m + std::log(sum);
  };
  const double lp = log_sum_exp(p_logits);
  const double lq = log_sum_exp(q_logits);
  double kl = 0.0;
  for (std::size_t i = 0; i < p_logits.size(); ++i) {
    const double log_p = p_logits[i] - lp;
    kl += std::exp(log_p) * (log_p - (q_logits[i] - lq));
  }
  return kl;
}

/// The LM head the model uses: embed_tokens itself when tied.
const Tensor& lm_head_of(const Format& format) {
  return format.weights().get(smollm2_config().tie_word_embeddings ? "model.embed_tokens.weight"
                                                                   : "lm_head.weight");
}

/// Teacher-forced: the post-final-norm hidden state of every position of
/// `ids`, from one prefill. Row t predicts token t + 1.
Tensor final_hidden_states(Model& model, std::span<const std::int32_t> ids) {
  const std::int64_t final_index = model.config().num_hidden_layers;
  Tensor out;
  model.reset();
  model.set_hidden_state_hook([&](std::int64_t index, const Tensor& hidden) {
    if (index == final_index) out = hidden.copy_contiguous();
  });
  model.forward(ids, 0);
  model.set_hidden_state_hook({});
  return out;
}

struct KlSet {
  double sum = 0.0;
  std::int64_t positions = 0;
  std::int64_t top1_agree = 0;
  double mean() const { return positions > 0 ? sum / static_cast<double>(positions) : 0.0; }
};

/// Adds every position of `ids` to `set`: KL(fp32 engine || Q8_0 engine) of
/// the next-token distributions. forward() returns only the last row's
/// logits, so each row's come from the hook's hidden state times that
/// engine's LM head, which is what forward() computes for the last row.
void add_kl(Model& f32, Model& q8, std::span<const std::int32_t> ids, KlSet& set) {
  const ModelConfig& cfg = smollm2_config();
  const Tensor& f32_head = lm_head_of(kF32);
  const Tensor& q8_head = lm_head_of(kQ8_0);
  const Tensor hf = final_hidden_states(f32, ids);
  const Tensor hq = final_hidden_states(q8, ids);

  const auto n = static_cast<std::int64_t>(ids.size());
  const std::int64_t vocab = cfg.vocab_size;
  constexpr std::int64_t kChunk = 32;  // rows of logits at a time: 6 MB per engine
  Tensor lf = Tensor::zeros(Shape{kChunk, vocab});
  Tensor lq = Tensor::zeros(Shape{kChunk, vocab});
  for (std::int64_t r0 = 0; r0 < n; r0 += kChunk) {
    const std::int64_t rows = std::min(kChunk, n - r0);
    Tensor out_f = lf.slice(0, 0, rows);
    Tensor out_q = lq.slice(0, 0, rows);
    ops::matmul(hf.slice(0, r0, rows), f32_head, out_f);
    ops::matmul(hq.slice(0, r0, rows), q8_head, out_q);
    for (std::int64_t r = 0; r < rows; ++r) {
      const std::span<const float> pf(lf.f32() + r * vocab, static_cast<std::size_t>(vocab));
      const std::span<const float> pq(lq.f32() + r * vocab, static_cast<std::size_t>(vocab));
      set.sum += kl_divergence(pf, pq);
      set.top1_agree += argmax_of(pf) == argmax_of(pq) ? 1 : 0;
      ++set.positions;
    }
  }
}

}  // namespace

// Release only: about 4000 teacher-forced positions through both engines.
// The budget is for the format and the quantizer, not for a kernel (kernel
// reassociation moves KL by ~1e-8), so a sanitizer build adds nothing.
TEST_CASE("SmolLM2 Q8_0: KL from the fp32 engine stays within the format's budget" *
          doctest::skip(!kRelease)) {
  if (skip_without_checkpoint()) return;
  std::vector<std::string> needed;
  for (int p = 0; p < kPrompts; ++p) {
    needed.push_back(prompt_file(p, "input_ids"));
    needed.push_back(prompt_file(p, "greedy_ids"));
  }
  MINILLM_SKIP_WITHOUT_ORACLE((needed));

  constexpr std::int64_t kLongest = 512;  // q8_0_eval_ids.h chunks are at most this long
  Model f32 = smollm2(kF32, kLongest);
  Model q8 = smollm2(kQ8_0, kLongest);

  // The harness first: the hook's last row times the LM head is the logits
  // forward() returns, in both engines.
  {
    const std::vector<std::int32_t> ids = prompt_ids(2);
    for (const Format* format : {&kF32, &kQ8_0}) {
      Model& m = format == &kF32 ? f32 : q8;
      const Tensor h = final_hidden_states(m, ids);
      Tensor row = Tensor::zeros(Shape{1, smollm2_config().vocab_size});
      ops::matmul(h.slice(0, h.dim(0) - 1, 1), lm_head_of(*format), row);
      m.reset();
      assert_allclose(span_of(row), m.forward(ids, 0), tol::kLogits,
                      std::string(format->name) + " teacher-forced last row");
    }
  }

  // The oracle prompts, each followed by fp32's 64 greedy tokens.
  KlSet oracle;
  for (int p = 0; p < kPrompts; ++p) {
    std::vector<std::int32_t> ids = prompt_ids(p);
    const std::vector<std::int32_t> greedy =
        oracle_npy(prompt_file(p, "greedy_ids")).as_i32_narrowing();
    ids.insert(ids.end(), greedy.begin(), greedy.end());
    add_kl(f32, q8, ids, oracle);
  }
  KlSet prose, code;
  for (const q8_0_eval::Sequence& seq : q8_0_eval::kSequences) {
    add_kl(f32, q8, seq.ids, seq.set == "code" ? code : prose);
  }

  struct Row {
    std::string name;
    const KlSet& set;
    double budget;
  };
  const tol::KlBudget& budget = tol::kQ8_0KlMean;
  for (const Row& r : {Row{"oracle prompts + continuations", oracle, budget.oracle_prompts},
                       Row{"prose", prose, budget.prose}, Row{"code", code, budget.code}}) {
    MESSAGE("q8_0 vs f32, " << r.name << ": " << r.set.positions << " positions, KL mean "
                            << r.set.mean() << " (budget " << r.budget << ", "
                            << 100.0 * r.set.mean() / r.budget << "%), top-1 agreement "
                            << 100.0 * static_cast<double>(r.set.top1_agree) /
                                   static_cast<double>(r.set.positions)
                            << "%");
    CHECK_MESSAGE(r.set.mean() <= r.budget, r.name << ": KL mean " << r.set.mean()
                                                   << " is over the budget " << r.budget);
  }
  CHECK(oracle.positions == 218);
  CHECK(prose.positions == 1060);
  CHECK(code.positions == 2748);
}
