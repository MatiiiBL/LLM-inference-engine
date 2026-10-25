// Workflow 06-attention-kv owns this file and test_kv_cache.cpp.
//
// ops::attention is checked three ways:
//   * against ref::attention on random inputs, across head layouts, odd
//     head_dims, strided cache views and contexts long enough to span many
//     key blocks — the cheap, exhaustive comparison;
//   * against PyTorch, on the three oracle cases and on two real layers of
//     the checkpoint, stage by stage (rmsnorm, q/k/v projections, RoPE,
//     attention, o_proj), so a drifting block can be bisected;
//   * through a KVCache, the way the model drives it: a prefill, then decode
//     steps and a continuation, each attending over cache.k(layer, len).

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "common/allclose.h"
#include "common/oracle.h"
#include "common/random.h"
#include "minillm/kv_cache.h"
#include "minillm/ops.h"
#include "minillm/runtime.h"

using namespace minillm;
using namespace minillm::test;

namespace {

/// An output buffer pre-filled with NaN. Any element the op fails to write
/// survives into the comparison and fails it, which proves "fully
/// overwritten" rather than assuming it.
Tensor nan_filled(const Shape& shape) {
  Tensor t = Tensor::empty(shape);
  float* p = t.f32();
  for (std::int64_t i = 0; i < t.numel(); ++i) p[i] = std::numeric_limits<float>::quiet_NaN();
  return t;
}

/// The kernel thread counts the threaded path is checked at. 10 is the
/// development machine's hardware_concurrency(); on a smaller machine it
/// oversubscribes, which exercises the split just as well.
constexpr int kThreadCounts[] = {1, 2, 3, 4, 10};

/// Calls run() at threads = 1, then at every other count in kThreadCounts,
/// and checks that each result is bitwise identical to the 1-thread one,
/// which it returns. Restores the thread count, pass or fail.
Tensor check_thread_invariant(const std::string& label, const std::function<Tensor()>& run) {
  struct Restore {
    int threads;
    ~Restore() { set_num_threads(threads); }
  } restore{num_threads()};
  const auto exact = [](float v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(v));
    return std::string(buf);
  };
  set_num_threads(1);
  Tensor single = run();
  for (const int threads : kThreadCounts) {
    if (threads == 1) continue;
    set_num_threads(threads);
    const Tensor multi = run();
    REQUIRE(multi.shape() == single.shape());
    const float* a = multi.f32();
    const float* b = single.f32();
    for (std::int64_t i = 0; i < single.numel(); ++i) {
      if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
        FAIL_CHECK(label << ": threads = " << threads << " differs from threads = 1 at index "
                         << i << ": " << exact(a[i]) << " vs " << exact(b[i]));
        break;
      }
    }
  }
  return single;
}


/// Runs ref:: and ops:: on the same inputs and compares the outputs.
void check_against_ref(const Tensor& q, const Tensor& k, const Tensor& v, std::int64_t start_pos,
                       float scale, const std::string& label) {
  Tensor expected = Tensor::zeros(q.shape());
  Tensor actual = nan_filled(q.shape());
  ref::attention(q, k, v, start_pos, scale, expected);
  ops::attention(q, k, v, start_pos, scale, actual);
  assert_allclose(actual, expected, tol::kAttention, "ops::attention vs ref " + label);
}

// --- real weights, without the loader --------------------------------------
//
// The projection stages need the checkpoint's weights, but the loader belongs
// to workflow 01 and tests/attention does not link it. A .safetensors file is
// an 8-byte little-endian header length, a flat JSON object mapping each name
// to {dtype, shape, data_offsets}, then the raw bytes, so pulling out a few
// named tensors takes a string search rather than a JSON parser. Test-only and
// deliberately strict: anything unexpected throws.

constexpr std::string_view kCheckpoint = "SmolLM2-135M/model.safetensors";

// SmolLM2-135M's config.json values, also recorded in oracle/data/manifest.json.
constexpr float kRmsNormEps = 1e-5f;
constexpr float kRopeTheta = 100000.0f;

class Checkpoint {
 public:
  explicit Checkpoint(const std::filesystem::path& file) : file_(file) {
    std::ifstream in(file_, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + file_.string());
    unsigned char len_bytes[8];
    in.read(reinterpret_cast<char*>(len_bytes), 8);
    std::uint64_t len = 0;
    for (int i = 7; i >= 0; --i) len = (len << 8) | len_bytes[i];
    header_.resize(static_cast<std::size_t>(len));
    in.read(header_.data(), static_cast<std::streamsize>(len));
    if (!in) throw std::runtime_error("truncated safetensors header in " + file_.string());
    data_start_ = 8 + len;
  }

  /// One tensor, converted to an owning contiguous F32 Tensor.
  Tensor get(const std::string& name) const {
    const std::string key = "\"" + name + "\"";
    const std::size_t at = header_.find(key);
    if (at == std::string::npos) throw std::runtime_error("no tensor " + name + " in checkpoint");
    const std::size_t open = header_.find('{', at);
    const std::size_t close = header_.find('}', open);
    const std::string_view entry = std::string_view(header_).substr(open, close - open + 1);

    const std::string dtype = string_field(entry, "dtype", name);
    const std::vector<std::int64_t> dims = int_list_field(entry, "shape", name);
    const std::vector<std::int64_t> offsets = int_list_field(entry, "data_offsets", name);
    if (dims.empty() || dims.size() > 4 || offsets.size() != 2) {
      throw std::runtime_error("unexpected shape or offsets for " + name);
    }

    const Shape shape(dims.data(), static_cast<int>(dims.size()));
    const std::size_t item = dtype == "BF16" ? 2 : dtype == "F32" ? 4 : 0;
    if (item == 0) throw std::runtime_error(name + " has unsupported dtype " + dtype);
    const auto n_bytes = static_cast<std::size_t>(offsets[1] - offsets[0]);
    if (n_bytes != static_cast<std::size_t>(shape.numel()) * item) {
      throw std::runtime_error(name + ": data_offsets do not match the shape");
    }

    std::vector<char> raw(n_bytes);
    std::ifstream in(file_, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(data_start_ + static_cast<std::uint64_t>(offsets[0])));
    in.read(raw.data(), static_cast<std::streamsize>(n_bytes));
    if (!in) throw std::runtime_error("truncated data for " + name);

    Tensor out = Tensor::empty(shape);
    float* dst = out.f32();
    if (item == 4) {
      std::memcpy(dst, raw.data(), n_bytes);
    } else {
      for (std::int64_t i = 0; i < shape.numel(); ++i) {
        std::uint16_t bits;
        std::memcpy(&bits, raw.data() + i * 2, 2);
        dst[i] = bf16_to_f32(bits);
      }
    }
    return out;
  }

 private:
  /// The text right after `"key":` inside one header entry.
  static std::string_view after_key(std::string_view entry, std::string_view key,
                                    const std::string& name) {
    const std::string quoted = "\"" + std::string(key) + "\"";
    std::size_t at = entry.find(quoted);
    if (at == std::string_view::npos) throw std::runtime_error(name + " has no " + quoted);
    at = entry.find(':', at + quoted.size());
    return entry.substr(at + 1);
  }

  static std::string string_field(std::string_view entry, std::string_view key,
                                  const std::string& name) {
    const std::string_view rest = after_key(entry, key, name);
    const std::size_t open = rest.find('"');
    const std::size_t close = rest.find('"', open + 1);
    return std::string(rest.substr(open + 1, close - open - 1));
  }

  static std::vector<std::int64_t> int_list_field(std::string_view entry, std::string_view key,
                                                  const std::string& name) {
    const std::string_view rest = after_key(entry, key, name);
    const std::size_t open = rest.find('[');
    const std::size_t close = rest.find(']', open);
    const std::string list(rest.substr(open + 1, close - open - 1));
    std::vector<std::int64_t> values;
    const char* p = list.c_str();
    while (*p != '\0') {
      char* end = nullptr;
      const long long v = std::strtoll(p, &end, 10);
      if (end == p) {
        ++p;  // a comma or whitespace
        continue;
      }
      values.push_back(v);
      p = end;
    }
    return values;
  }

  std::filesystem::path file_;
  std::string header_;
  std::uint64_t data_start_ = 0;
};

/// Like MINILLM_SKIP_WITHOUT_ORACLE, for the checkpoint. Returns false (after
/// recording the skip) if it is missing.
bool have_checkpoint() {
  const std::filesystem::path file = models_dir() / kCheckpoint;
  std::error_code ec;
  if (std::filesystem::is_regular_file(file, ec)) return true;
  const std::string reason =
      "missing checkpoint: " + file.string() + " — run `python oracle/generate.py`";
  note_skip(reason);
  MESSAGE("SKIP: " << reason);
  return false;
}

}  // namespace

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

TEST_CASE("ops::attention is bitwise identical at every thread count") {
  // Threads split (query row, KV head) pairs; each pair's output is computed
  // start to finish by one thread, exactly as on one.
  struct Case {
    const char* name;
    std::int64_t n_tokens;
    std::int64_t n_q_heads;
    std::int64_t n_kv_heads;
    std::int64_t start_pos;
  };
  const Case cases[] = {
      {"prefill", 128, 9, 3, 0},
      {"decode at 191", 1, 9, 3, 191},
      {"1 token, 1 key", 1, 9, 3, 0},
      {"7 tokens after 30", 7, 9, 3, 30},
      {"multi-head, 7 tokens", 7, 4, 4, 5},
      {"decode at 49151", 1, 9, 3, 49151},
  };
  const float scale = 1.0f / 8.0f;
  for (const Case& c : cases) {
    const std::string label = std::string("ops::attention ") + c.name;
    const std::int64_t n_kv = c.start_pos + c.n_tokens;
    const Tensor q = random_tensor({c.n_tokens, c.n_q_heads, 64}, 411);
    const Tensor k = random_tensor({n_kv, c.n_kv_heads, 64}, 412);
    const Tensor v = random_tensor({n_kv, c.n_kv_heads, 64}, 413);
    const Tensor single = check_thread_invariant(label, [&] {
      Tensor out = nan_filled(q.shape());
      ops::attention(q, k, v, c.start_pos, scale, out);
      return out;
    });
    Tensor expected = Tensor::zeros(q.shape());
    ref::attention(q, k, v, c.start_pos, scale, expected);
    assert_allclose(single, expected, tol::kAttention, label + " vs ref");
  }
}

TEST_CASE("ops::attention agrees with ref::attention across head layouts and lengths") {
  // GQA group sizes 1 (multi-head) through 12, head_dims that are not a
  // multiple of any SIMD width, and contexts from a single key to ~1000 keys.
  struct Case {
    const char* name;
    std::int64_t n_tokens;
    std::int64_t n_q_heads;
    std::int64_t n_kv_heads;
    std::int64_t head_dim;
    std::int64_t start_pos;
    std::int64_t n_kv;
  };
  const Case cases[] = {
      {"single key", 1, 9, 3, 64, 0, 1},
      {"prefill 37", 37, 9, 3, 64, 0, 37},
      {"decode at 1000", 1, 9, 3, 64, 1000, 1001},
      {"continuation at 300, cache longer than needed", 5, 9, 3, 64, 300, 320},
      {"multi-head, group 1", 6, 4, 4, 64, 3, 9},
      {"multi-query, one KV head", 4, 8, 1, 64, 130, 134},
      {"group 9", 3, 18, 2, 32, 140, 143},
      {"group 12", 2, 12, 1, 16, 7, 9},
      {"head_dim 1", 3, 2, 1, 1, 2, 5},
      {"head_dim 5", 4, 6, 2, 5, 0, 4},
      {"head_dim 17", 3, 9, 3, 17, 129, 140},
      {"head_dim 80", 3, 6, 2, 80, 140, 150},
      {"head_dim 128", 2, 8, 2, 128, 255, 257},
  };

  std::uint64_t seed = 410;
  for (const Case& c : cases) {
    const Tensor q = random_tensor({c.n_tokens, c.n_q_heads, c.head_dim}, seed++);
    const Tensor k = random_tensor({c.n_kv, c.n_kv_heads, c.head_dim}, seed++);
    const Tensor v = random_tensor({c.n_kv, c.n_kv_heads, c.head_dim}, seed++);
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
    check_against_ref(q, k, v, c.start_pos, scale, c.name);
  }
}

TEST_CASE("ops::attention reads strided and offset KV views") {
  // Two layouts KVCache never hands out but the contract allows: one KV head
  // sliced out of a wider buffer (non-contiguous), and a slice that does not
  // start at the buffer's first head.
  const std::int64_t n_kv = 140;
  const Tensor wide_k = random_tensor({n_kv, 6, 64}, 430);
  const Tensor wide_v = random_tensor({n_kv, 6, 64}, 431);
  const Tensor q = random_tensor({3, 9, 64}, 432);

  for (const std::int64_t first_head : {0, 2, 3}) {
    const Tensor k = wide_k.slice(1, first_head, 3);
    const Tensor v = wide_v.slice(1, first_head, 3);
    REQUIRE_FALSE(k.is_contiguous());
    check_against_ref(q, k, v, 130, 0.125f,
                      "kv heads [" + std::to_string(first_head) + ", " +
                          std::to_string(first_head + 3) + ") of 6");
  }

  // One layer's prefix of a [layer, pos, kv_head, head_dim] buffer, with a
  // single KV head picked out of it: strided along both outer dimensions.
  const Tensor cache = random_tensor({2, 64, 3, 64}, 433);
  const Tensor k = cache.row(1).slice(1, 1, 1);
  const Tensor v = cache.row(0).slice(1, 2, 1);
  REQUIRE_FALSE(k.is_contiguous());
  check_against_ref(random_tensor({2, 4, 64}, 434), k, v, 40, 0.125f, "single head of a layer");
}

TEST_CASE("ops::attention never reads past the causal prefix") {
  // Positions beyond start_pos + n_tokens - 1 are poisoned with NaN. Causality
  // is a loop bound, not a mask: a masked read would turn 0 * NaN into NaN.
  const std::int64_t n_tokens = 4;
  const std::int64_t start_pos = 150;
  const std::int64_t needed = start_pos + n_tokens;
  const std::int64_t n_kv = needed + 40;
  Tensor k = random_tensor({n_kv, 3, 64}, 440);
  Tensor v = random_tensor({n_kv, 3, 64}, 441);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (std::int64_t i = needed * 3 * 64; i < n_kv * 3 * 64; ++i) {
    k.f32()[i] = nan;
    v.f32()[i] = nan;
  }
  const Tensor q = random_tensor({n_tokens, 9, 64}, 442);
  check_against_ref(q, k, v, start_pos, 0.125f, "NaN past the prefix");

  // Within one pass, row t must not see keys of rows after it: each row of a
  // prefill equals a one-token decode at its own position over its own prefix.
  Tensor prefill = nan_filled(q.shape());
  ops::attention(q, k, v, start_pos, 0.125f, prefill);
  for (std::int64_t t = 0; t < n_tokens; ++t) {
    const std::int64_t len = start_pos + t + 1;
    Tensor decode = nan_filled({1, 9, 64});
    ops::attention(q.slice(0, t, 1), k.slice(0, 0, len), v.slice(0, 0, len), start_pos + t,
                   0.125f, decode);
    assert_allclose(decode, prefill.slice(0, t, 1), tol::kAttention,
                    "prefill row " + std::to_string(t) + " vs decode at " +
                        std::to_string(start_pos + t));
  }
}

TEST_CASE("ops::attention keeps the softmax stable for large logits") {
  // Large scores are ill-conditioned in float: near 100 one ulp is ~1e-5, and
  // softmax turns an absolute score error into the same relative weight
  // error, so two correct kernels that merely sum a dot product in a
  // different order disagree by ~1e-4. Every score here is therefore exact
  // in any summation order — small-integer q and k, power-of-two scales —
  // which leaves only the softmax itself to compare.
  const std::int64_t n_kv = 600;
  auto small_ints = [](const Shape& shape, std::uint64_t seed) {
    Tensor t = random_tensor_uniform(shape, seed, -3.5f, 3.5f);
    for (std::int64_t i = 0; i < t.numel(); ++i) t.f32()[i] = std::round(t.f32()[i]);
    return t;
  };
  const Tensor v = random_tensor({n_kv, 3, 64}, 452);

  // Scores up to ~+-100: exp() of an unshifted score overflows float.
  check_against_ref(small_ints({2, 9, 64}, 450), small_ints({n_kv, 3, 64}, 451), v, n_kv - 2,
                    0.25f, "scores up to ~100");

  // Scores 0.25 * j, rising with position, so the running max rises in every
  // block and everything accumulated so far must be rescaled; and the mirror
  // image, where the first key dominates and later ones underflow to zero.
  // Only dimension 0 of each key is non-zero, so the dot product is exact.
  Tensor ones = Tensor::empty({1, 9, 64});
  for (std::int64_t i = 0; i < ones.numel(); ++i) ones.f32()[i] = 1.0f;
  Tensor rising = Tensor::zeros({n_kv, 3, 64});
  Tensor falling = Tensor::zeros({n_kv, 3, 64});
  for (std::int64_t j = 0; j < n_kv; ++j) {
    for (std::int64_t g = 0; g < 3; ++g) {
      rising.f32()[rising.offset(j, g, 0)] = 0.25f * static_cast<float>(j);
      falling.f32()[falling.offset(j, g, 0)] = 0.25f * static_cast<float>(n_kv - j);
    }
  }
  check_against_ref(ones, rising, v, n_kv - 1, 1.0f, "rising scores");
  check_against_ref(ones, falling, v, n_kv - 1, 1.0f, "falling scores");
}

TEST_CASE("ops::attention accepts an empty pass") {
  // Zero query rows is a legal no-op, including against an empty cache.
  const Tensor q = Tensor::zeros({0, 9, 64});
  Tensor out = Tensor::zeros({0, 9, 64});
  CHECK_NOTHROW(ops::attention(q, Tensor::zeros({0, 3, 64}), Tensor::zeros({0, 3, 64}), 0,
                               0.125f, out));
  CHECK_NOTHROW(ops::attention(q, Tensor::zeros({5, 3, 64}), Tensor::zeros({5, 3, 64}), 5,
                               0.125f, out));
}

TEST_CASE("ops::attention validates its arguments") {
  const Tensor q = Tensor::zeros({2, 4, 8});
  const Tensor k = Tensor::zeros({4, 2, 8});
  Tensor out = Tensor::zeros({2, 4, 8});

  // Cache too short for start_pos + n_tokens.
  CHECK_THROWS_AS(ops::attention(q, k, k, 3, 1.0f, out), std::invalid_argument);
  // n_q_heads not a multiple of n_kv_heads.
  const Tensor k3 = Tensor::zeros({4, 3, 8});
  CHECK_THROWS_AS(ops::attention(q, k3, k3, 0, 1.0f, out), std::invalid_argument);
  // head_dim mismatch between q and the cache.
  const Tensor k_wide = Tensor::zeros({4, 2, 16});
  CHECK_THROWS_AS(ops::attention(q, k_wide, k_wide, 0, 1.0f, out), std::invalid_argument);
  // out shape must equal q shape.
  Tensor out_wrong = Tensor::zeros({2, 4, 4});
  CHECK_THROWS_AS(ops::attention(q, k, k, 0, 1.0f, out_wrong), std::invalid_argument);
  // negative start_pos.
  CHECK_THROWS_AS(ops::attention(q, k, k, -1, 1.0f, out), std::invalid_argument);

  // Beyond the reference's list: K and V must agree, q and out must be
  // contiguous f32 of rank 3, and a zero-head cache is rejected.
  const Tensor k_long = Tensor::zeros({5, 2, 8});
  CHECK_THROWS_AS(ops::attention(q, k, k_long, 0, 1.0f, out), std::invalid_argument);
  const Tensor q_strided = Tensor::zeros({2, 4, 16}).slice(2, 0, 8);
  REQUIRE_FALSE(q_strided.is_contiguous());
  CHECK_THROWS_AS(ops::attention(q_strided, k, k, 0, 1.0f, out), std::invalid_argument);
  Tensor out_strided = Tensor::zeros({2, 4, 16}).slice(2, 0, 8);
  CHECK_THROWS_AS(ops::attention(q, k, k, 0, 1.0f, out_strided), std::invalid_argument);
  const Tensor q_bf16 = Tensor::zeros({2, 4, 8}, DType::BF16);
  CHECK_THROWS_AS(ops::attention(q_bf16, k, k, 0, 1.0f, out), std::invalid_argument);
  const Tensor q_rank2 = Tensor::zeros({2, 32});
  CHECK_THROWS_AS(ops::attention(q_rank2, k, k, 0, 1.0f, out), std::invalid_argument);
  const Tensor k_no_heads = Tensor::zeros({4, 0, 8});
  CHECK_THROWS_AS(ops::attention(q, k_no_heads, k_no_heads, 0, 1.0f, out),
                  std::invalid_argument);
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

TEST_CASE("KVCache round trip: prefill, decode and continuation match raw tensors") {
  // Drives the cache exactly the way Model::forward does — write every new
  // position of every layer, set_seq_len() once per pass, attend over
  // cache.k(layer, len) — and checks each pass against the same computation
  // on plain tensors that hold the full history. Layer 0 is written with
  // store(), layer 1 in place through k_slot()/v_slot(), and the layers get
  // different data so a layer mix-up shows.
  const std::int64_t n_layers = 2;
  const std::int64_t max_seq = 48;
  const std::int64_t n_q_heads = 9;
  const std::int64_t n_kv_heads = 3;
  const std::int64_t head_dim = 64;
  const std::int64_t slot = n_kv_heads * head_dim;
  KVCache cache(n_layers, max_seq, n_kv_heads, head_dim);

  std::vector<Tensor> k_hist;
  std::vector<Tensor> v_hist;
  for (std::int64_t l = 0; l < n_layers; ++l) {
    k_hist.push_back(Tensor::zeros({max_seq, n_kv_heads, head_dim}));
    v_hist.push_back(Tensor::zeros({max_seq, n_kv_heads, head_dim}));
  }

  std::uint64_t seed = 460;
  auto run_pass = [&](std::int64_t n_tokens, const std::string& label) {
    const std::int64_t start_pos = cache.seq_len();
    const std::int64_t len = start_pos + n_tokens;
    for (std::int64_t l = 0; l < n_layers; ++l) {
      const Tensor k_new = random_tensor({n_tokens, n_kv_heads, head_dim}, seed++);
      const Tensor v_new = random_tensor({n_tokens, n_kv_heads, head_dim}, seed++);
      for (std::int64_t t = 0; t < n_tokens; ++t) {
        const float* k_row = k_new.f32() + t * slot;
        const float* v_row = v_new.f32() + t * slot;
        if (l == 0) {
          cache.store(l, start_pos + t, k_row, v_row);
        } else {
          std::memcpy(cache.k_slot(l, start_pos + t), k_row, sizeof(float) * slot);
          std::memcpy(cache.v_slot(l, start_pos + t), v_row, sizeof(float) * slot);
        }
        std::memcpy(k_hist[l].f32() + (start_pos + t) * slot, k_row, sizeof(float) * slot);
        std::memcpy(v_hist[l].f32() + (start_pos + t) * slot, v_row, sizeof(float) * slot);
      }
    }
    cache.set_seq_len(len);

    for (std::int64_t l = 0; l < n_layers; ++l) {
      const Tensor q = random_tensor({n_tokens, n_q_heads, head_dim}, seed++);
      Tensor expected = Tensor::zeros(q.shape());
      Tensor actual = nan_filled(q.shape());
      ref::attention(q, k_hist[l].slice(0, 0, len), v_hist[l].slice(0, 0, len), start_pos, 0.125f,
                     expected);
      ops::attention(q, cache.k(l, cache.seq_len()), cache.v(l, cache.seq_len()), start_pos,
                     0.125f, actual);
      assert_allclose(actual, expected, tol::kAttention,
                      label + " layer " + std::to_string(l) + " at " + std::to_string(start_pos));
    }
  };

  run_pass(7, "prefill");
  for (int i = 0; i < 5; ++i) run_pass(1, "decode");
  run_pass(4, "continuation");
  CHECK(cache.seq_len() == 16);

  // A second sequence after reset(): the first one's 16 positions are still
  // in memory, and a shorter sequence must not see them. Poison them so a
  // stray read cannot go unnoticed.
  cache.reset();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (std::int64_t l = 0; l < n_layers; ++l) {
    for (std::int64_t p = 0; p < 16; ++p) {
      std::fill_n(cache.k_slot(l, p), slot, nan);
      std::fill_n(cache.v_slot(l, p), slot, nan);
    }
  }
  run_pass(3, "prefill after reset");
  run_pass(1, "decode after reset");
  CHECK(cache.seq_len() == 4);
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

TEST_CASE("a real layer decoded token by token through a KVCache matches transformers") {
  // transformers ran the 13-token prompt in one pass. Here the first 9 tokens
  // are a prefill and the last 4 are decode steps, each attending over the
  // cache — the engine's incremental path must land on the same numbers.
  for (const std::string layer : {"l0", "l15"}) {
    const std::string base = "attention/" + layer + "_";
    MINILLM_SKIP_WITHOUT_ORACLE({base + "q_rope.npy", base + "k_rope.npy", base + "v_proj.npy",
                                 base + "attn_out.npy"});

    const Tensor q = oracle_npy(base + "q_rope.npy").to_tensor();
    const Tensor k = oracle_npy(base + "k_rope.npy").to_tensor();
    const Tensor v = oracle_npy(base + "v_proj.npy").to_tensor();
    const Tensor expected = oracle_npy(base + "attn_out.npy").to_tensor();
    const std::int64_t n = q.shape()[0];
    const std::int64_t n_kv_heads = k.shape()[1];
    const std::int64_t head_dim = k.shape()[2];
    const std::int64_t slot = n_kv_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const std::int64_t n_prefill = 9;
    REQUIRE(n > n_prefill);

    KVCache cache(1, n, n_kv_heads, head_dim);
    Tensor actual = nan_filled(expected.shape());
    for (std::int64_t t = 0; t < n_prefill; ++t) {
      cache.store(0, t, k.f32() + t * slot, v.f32() + t * slot);
    }
    cache.set_seq_len(n_prefill);
    Tensor prefill_out = actual.slice(0, 0, n_prefill);
    ops::attention(q.slice(0, 0, n_prefill), cache.k(0, n_prefill), cache.v(0, n_prefill), 0,
                   scale, prefill_out);

    for (std::int64_t t = n_prefill; t < n; ++t) {
      cache.store(0, t, k.f32() + t * slot, v.f32() + t * slot);
      cache.set_seq_len(t + 1);
      Tensor step_out = actual.slice(0, t, 1);
      ops::attention(q.slice(0, t, 1), cache.k(0, t + 1), cache.v(0, t + 1), t, scale, step_out);
    }
    assert_allclose(actual, expected, tol::kAttention, "incremental attention " + layer);
  }
}

TEST_CASE("attention block stages for a real layer match transformers") {
  // The rest of the block around the op, one stage at a time and each fed the
  // golden input of that stage, so a drift is pinned to the stage that caused
  // it: input_layernorm, the q/k/v projections, RoPE, and o_proj. Then the
  // whole block chained from hidden_in, the way the model will run it.
  if (!have_checkpoint()) return;
  const Checkpoint ckpt(models_dir() / kCheckpoint);

  for (const std::string layer : {"l0", "l15"}) {
    const std::string base = "attention/" + layer + "_";
    MINILLM_SKIP_WITHOUT_ORACLE(
        {"attention/positions.npy", base + "hidden_in.npy", base + "normed.npy",
         base + "q_proj.npy", base + "k_proj.npy", base + "v_proj.npy", base + "q_rope.npy",
         base + "k_rope.npy", base + "attn_out.npy", base + "o_proj.npy"});

    const std::string prefix = "model.layers." + layer.substr(1) + ".";
    const Tensor ln_w = ckpt.get(prefix + "input_layernorm.weight");
    const Tensor wq = ckpt.get(prefix + "self_attn.q_proj.weight");
    const Tensor wk = ckpt.get(prefix + "self_attn.k_proj.weight");
    const Tensor wv = ckpt.get(prefix + "self_attn.v_proj.weight");
    const Tensor wo = ckpt.get(prefix + "self_attn.o_proj.weight");

    const std::vector<std::int64_t> positions = oracle_npy("attention/positions.npy").as_i64();
    const Tensor hidden_in = oracle_npy(base + "hidden_in.npy").to_tensor();
    const Tensor normed = oracle_npy(base + "normed.npy").to_tensor();
    const Tensor q_proj = oracle_npy(base + "q_proj.npy").to_tensor();
    const Tensor k_proj = oracle_npy(base + "k_proj.npy").to_tensor();
    const Tensor v_proj = oracle_npy(base + "v_proj.npy").to_tensor();
    const Tensor q_rope = oracle_npy(base + "q_rope.npy").to_tensor();
    const Tensor k_rope = oracle_npy(base + "k_rope.npy").to_tensor();
    const Tensor attn_out = oracle_npy(base + "attn_out.npy").to_tensor();
    const Tensor o_proj = oracle_npy(base + "o_proj.npy").to_tensor();

    const std::int64_t n = hidden_in.shape()[0];
    const Shape q_heads = q_rope.shape();
    const Shape k_heads = k_rope.shape();
    const float scale = 1.0f / std::sqrt(static_cast<float>(q_heads[2]));
    const Tolerance rope_tol = rope_tolerance(positions.back());

    // input_layernorm
    Tensor x = Tensor::zeros(normed.shape());
    ops::rmsnorm(hidden_in, ln_w, kRmsNormEps, x);
    assert_allclose(x, normed, tol::kReduction, layer + " input_layernorm");

    // q/k/v projections, from the golden normed input
    Tensor q = Tensor::zeros(q_proj.shape());
    Tensor k = Tensor::zeros(k_proj.shape());
    Tensor v = Tensor::zeros(v_proj.shape());
    ops::matmul(normed, wq, q);
    ops::matmul(normed, wk, k);
    ops::matmul(normed, wv, v);
    assert_allclose(q, q_proj, tol::kProjection, layer + " q_proj");
    assert_allclose(k, k_proj, tol::kProjection, layer + " k_proj");
    assert_allclose(v, v_proj, tol::kProjection, layer + " v_proj");

    // RoPE, from the golden projections
    Tensor q_rot = q_proj.copy_contiguous().reshape(q_heads);
    Tensor k_rot = k_proj.copy_contiguous().reshape(k_heads);
    ops::rope(q_rot, &k_rot, positions, kRopeTheta);
    assert_allclose(q_rot, q_rope, rope_tol, layer + " q after RoPE");
    assert_allclose(k_rot, k_rope, rope_tol, layer + " k after RoPE");

    // o_proj, from the golden attention output
    Tensor o = Tensor::zeros(o_proj.shape());
    ops::matmul(attn_out.reshape({n, q_heads[1] * q_heads[2]}), wo, o);
    assert_allclose(o, o_proj, tol::kProjection, layer + " o_proj");

    // The whole block from hidden_in, nothing golden in between.
    ops::rmsnorm(hidden_in, ln_w, kRmsNormEps, x);
    ops::matmul(x, wq, q);
    ops::matmul(x, wk, k);
    ops::matmul(x, wv, v);
    Tensor q3 = q.reshape(q_heads);
    Tensor k3 = k.reshape(k_heads);
    ops::rope(q3, &k3, positions, kRopeTheta);
    Tensor a = nan_filled(q_heads);
    ops::attention(q3, k3, v.reshape(k_heads), 0, scale, a);
    ops::matmul(a.reshape({n, q_heads[1] * q_heads[2]}), wo, o);
    assert_allclose(o, o_proj, tol::kProjection, layer + " whole block, chained");
  }
}
