// minillm benchmarks — owned by workflow 10-bench-ci.
//
// Deliberately no benchmarking framework: timing a kernel is a loop and a
// clock, and the dependency list in docs/ARCHITECTURE.md §6 is closed.
//
// What it measures, meaningful on the `release` preset only (the binary
// refuses to run from a sanitizer or unoptimized build):
//   * every op in ops.h at the model's real shapes, ops:: against ref::, so
//     the speedup of each optimization is visible and attributable
//   * prefill and decode tokens/s for the whole model
//   * peak resident memory
//
// Each measurement prints a row of a human-readable table as it finishes, and
// the run ends with one machine-readable line per measurement:
//
//   BENCH {"name":"op.prefill.matmul.q_proj.ops","value":812.4,"unit":"us",...}
//
// The README section "Reading the benchmark output" explains both.
//
// Exit status: 0 when everything requested was measured or skipped as not
// implemented yet, 1 on any error, 2 when the model directory is missing.

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "minillm/config.h"
#include "minillm/model.h"
#include "minillm/ops.h"
#include "minillm/tensor.h"

// Numbers from a sanitizer build are meaningless, and quoting them is worse
// than not measuring at all, so detect one and refuse to run.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define MINILLM_BENCH_SANITIZED 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define MINILLM_BENCH_SANITIZED 1
#endif
#endif
#ifndef MINILLM_BENCH_SANITIZED
#define MINILLM_BENCH_SANITIZED 0
#endif

namespace {

namespace fs = std::filesystem;
using namespace minillm;
using Clock = std::chrono::steady_clock;

constexpr int kExitError = 1;
constexpr int kExitNoModel = 2;

/// Why this binary's numbers would be meaningless, or nullptr for a release
/// build.
constexpr const char* kBuildProblem =
#if MINILLM_BENCH_SANITIZED
    "built with a sanitizer";
#elif !defined(__OPTIMIZE__)
    "built without optimization";
#elif !defined(NDEBUG)
    "built without NDEBUG";
#else
    nullptr;
#endif

constexpr const char* kCompiler =
#if defined(__clang__)
    "clang " __clang_version__;
#elif defined(__GNUC__)
    "gcc " __VERSION__;
#else
    "unknown compiler";
#endif

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

struct Options {
  fs::path model_dir = "models/SmolLM2-135M";
  std::string filter;
  int samples = 5;
  double min_sample_s = 0.05;
  double warmup_s = 0.05;
  std::int64_t prompt_tokens = 128;
  std::int64_t decode_tokens = 64;
  bool allow_debug_build = false;
};

constexpr const char* kUsage = R"(usage: minillm_bench [options]

Benchmarks every op in ops.h (ops:: vs ref::) at the model's real shapes, then
prefill and decode throughput for the whole model, then peak resident memory.
Run it from the repository root on the release preset:

  cmake --preset release && cmake --build --preset release
  ./build/release/bench/minillm_bench

options:
  --model-dir DIR       checkpoint directory (default: models/SmolLM2-135M)
  --filter TEXT         only run benchmarks whose name contains TEXT,
                        e.g. "matmul", "decode", "model."
  --samples N           timed samples per measurement, >= 3 (default: 5)
  --min-time-ms MS      minimum duration of one op sample (default: 50)
  --prompt-tokens N     prefill length (default: 128)
  --decode-tokens N     tokens generated after the prompt (default: 64)
  --allow-debug-build   run even from a sanitizer or unoptimized build, to
                        smoke-test this program; its numbers are meaningless
  -h, --help            this text

exit status: 0 ok, 1 error, 2 model directory missing
)";

std::int64_t parse_int(std::string_view flag, const char* text, std::int64_t min) {
  char* end = nullptr;
  const long long v = std::strtoll(text, &end, 10);
  if (end == text || *end != '\0' || v < min) {
    throw std::invalid_argument(std::string(flag) + " needs an integer >= " +
                                std::to_string(min) + ", got '" + text + "'");
  }
  return v;
}

/// Returns the options, or an exit status if the program should stop.
std::optional<int> parse_options(int argc, char** argv, Options& opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto value = [&]() -> const char* {
      if (i + 1 >= argc) throw std::invalid_argument(std::string(arg) + " needs a value");
      return argv[++i];
    };
    if (arg == "-h" || arg == "--help") {
      std::fputs(kUsage, stdout);
      return 0;
    } else if (arg == "--model-dir") {
      opt.model_dir = value();
    } else if (arg == "--filter") {
      opt.filter = value();
    } else if (arg == "--samples") {
      // One sample has no spread, and with two the median is just the mean.
      opt.samples = static_cast<int>(parse_int(arg, value(), 3));
    } else if (arg == "--min-time-ms") {
      opt.min_sample_s = static_cast<double>(parse_int(arg, value(), 1)) / 1000.0;
    } else if (arg == "--prompt-tokens") {
      opt.prompt_tokens = parse_int(arg, value(), 1);
    } else if (arg == "--decode-tokens") {
      opt.decode_tokens = parse_int(arg, value(), 1);
    } else if (arg == "--allow-debug-build") {
      opt.allow_debug_build = true;
    } else {
      throw std::invalid_argument("unknown option '" + std::string(arg) + "' (see --help)");
    }
  }
  return std::nullopt;
}

bool selected(const Options& opt, std::string_view name) {
  return opt.filter.empty() || name.find(opt.filter) != std::string_view::npos;
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

[[gnu::format(printf, 1, 2)]] std::string strprintf(const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  return buf;
}

/// "[128,576]" — compact, for the shape column.
std::string dims(std::initializer_list<std::int64_t> extents) {
  std::string s = "[";
  for (const std::int64_t e : extents) {
    if (s.size() > 1) s += ',';
    s += std::to_string(e);
  }
  return s + "]";
}

std::string human_time(double seconds) {
  if (seconds < 1e-6) return strprintf("%7.1f ns", seconds * 1e9);
  if (seconds < 1e-3) return strprintf("%7.2f us", seconds * 1e6);
  if (seconds < 1.0) return strprintf("%7.2f ms", seconds * 1e3);
  return strprintf("%7.3f s ", seconds);
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

/// Tells the compiler the memory behind `p` is read and written here, so the
/// stores that produce a timed result cannot be deleted as dead. The ops live
/// in another translation unit, so nothing could be optimized away today
/// anyway; this keeps the numbers honest if LTO is ever turned on.
inline void keep_alive(const void* p) { __asm__ __volatile__("" : : "r"(p) : "memory"); }

/// Sink for scalar results (argmax), for the same reason.
volatile std::int32_t g_sink = 0;

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/// Summary of several samples of one quantity.
class Stats {
 public:
  explicit Stats(std::vector<double> values) : values_(std::move(values)) {
    std::sort(values_.begin(), values_.end());
  }
  int count() const { return static_cast<int>(values_.size()); }
  double min() const { return values_.front(); }
  double max() const { return values_.back(); }
  double median() const {
    const std::size_t n = values_.size();
    return n % 2 == 1 ? values_[n / 2] : (values_[n / 2 - 1] + values_[n / 2]) / 2.0;
  }
  /// Half the range, relative to the median: the "+/-" column.
  double spread() const { return (max() - min()) / 2.0 / median(); }

 private:
  std::vector<double> values_;
};

/// Per-call times of a repeatedly called function.
struct Timing {
  Stats per_call;                   ///< seconds per call, one value per sample
  std::int64_t calls_per_sample;    ///< calls batched into each sample
  std::int64_t iterations() const { return calls_per_sample * per_call.count(); }
  std::string calls() const {
    return std::to_string(per_call.count()) + "x" + std::to_string(calls_per_sample);
  }
};

/// Warms `fn` up and returns how many calls to batch into one sample. A fast
/// call is batched so that each sample lasts at least opt.min_sample_s, which
/// buries the clock's resolution and overhead.
std::int64_t calibrate(const std::function<void()>& fn, const Options& opt) {
  // The warm-up absorbs first-touch page faults on freshly allocated outputs,
  // lazily built tables (rope may cache its frequencies), and the CPU's clock
  // ramp. It also estimates the cost of one call.
  std::int64_t warm_calls = 0;
  const Clock::time_point warm_start = Clock::now();
  do {
    fn();
    ++warm_calls;
  } while (seconds_since(warm_start) < opt.warmup_s);
  const double estimate = seconds_since(warm_start) / static_cast<double>(warm_calls);
  return std::max<std::int64_t>(1, std::llround(std::ceil(opt.min_sample_s / estimate)));
}

/// Seconds per call over one sample of `calls` calls.
double time_sample(const std::function<void()>& fn, std::int64_t calls) {
  const Clock::time_point start = Clock::now();
  for (std::int64_t c = 0; c < calls; ++c) fn();
  return seconds_since(start) / static_cast<double>(calls);
}

/// Two functions timed against each other.
struct Comparison {
  Timing a;
  Timing b;
  Stats speedup;  ///< b / a for each interleaved pair of samples
};

/// Times two functions with their samples interleaved, a b a b ..., and takes
/// the speedup pair by pair, so that drift in the machine's state — clock
/// frequency, thermals, a background process — lands on both sides of each
/// ratio instead of skewing the comparison. Timed back to back instead, two
/// identical functions differ by up to ~20% on a busy machine.
Comparison time_interleaved(const std::function<void()>& a, const std::function<void()>& b,
                            const Options& opt) {
  const std::int64_t calls_a = calibrate(a, opt);
  const std::int64_t calls_b = calibrate(b, opt);
  std::vector<double> per_call_a;
  std::vector<double> per_call_b;
  std::vector<double> ratio;
  for (int s = 0; s < opt.samples; ++s) {
    per_call_a.push_back(time_sample(a, calls_a));
    per_call_b.push_back(time_sample(b, calls_b));
    ratio.push_back(per_call_b.back() / per_call_a.back());
  }
  return {Timing{Stats(std::move(per_call_a)), calls_a}, Timing{Stats(std::move(per_call_b)), calls_b},
          Stats(std::move(ratio))};
}

/// One result, as printed on a machine-readable line.
struct Measurement {
  std::string name;
  double value = 0.0;
  std::string unit;
  std::int64_t iterations = 0;
  /// Spread over samples, in `unit`. samples == 0 for a value that is not
  /// sampled, such as peak RSS.
  int samples = 0;
  double min = 0.0;
  double max = 0.0;
};

/// Collects measurements and skips, and prints the machine-readable lines.
class Report {
 public:
  void add(Measurement m) { results_.push_back(std::move(m)); }

  /// A value that is not sampled, such as peak RSS.
  void add_value(std::string name, double value, std::string unit, std::int64_t iterations) {
    add({std::move(name), value, std::move(unit), iterations, 0, 0.0, 0.0});
  }

  void add_time(std::string name, const Timing& t) {
    add({std::move(name), t.per_call.median() * 1e6, "us", t.iterations(), t.per_call.count(),
         t.per_call.min() * 1e6, t.per_call.max() * 1e6});
  }

  void skip(std::string what, std::string why) {
    std::printf("SKIP %s: %s\n", what.c_str(), why.c_str());
    std::fflush(stdout);
    skips_.push_back(std::move(what) + ": " + std::move(why));
  }

  void print_machine_lines() const {
    for (const Measurement& m : results_) {
      std::string line = "BENCH {\"name\":\"" + m.name + "\"" + strprintf(",\"value\":%.6g", m.value) +
                         ",\"unit\":\"" + m.unit + "\",\"iterations\":" + std::to_string(m.iterations);
      if (m.samples > 0) {
        line += ",\"samples\":" + std::to_string(m.samples) +
                strprintf(",\"min\":%.6g,\"max\":%.6g", m.min, m.max);
      }
      std::printf("%s}\n", line.c_str());
    }
  }

  void print_summary() const {
    std::printf("\n%zu measurement(s), %zu skipped\n", results_.size(), skips_.size());
    for (const std::string& s : skips_) std::printf("  skipped %s\n", s.c_str());
    std::printf("\n");
  }

 private:
  std::vector<Measurement> results_;
  std::vector<std::string> skips_;
};

/// Peak resident set size of this process so far, in MiB.
double peak_rss_mib() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
  return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);  // bytes on macOS
#else
  return static_cast<double>(usage.ru_maxrss) / 1024.0;  // kilobytes on Linux
#endif
}

/// The stubs throw std::logic_error("not implemented: ..."). Those mean "not
/// written yet" and are reported as skips; any other exception is a failure.
bool is_not_implemented(const std::logic_error& e) {
  return std::string_view(e.what()).starts_with("not implemented");
}

// ---------------------------------------------------------------------------
// Model config
// ---------------------------------------------------------------------------

/// The value of a top-level numeric key in a flat JSON document, if present.
std::optional<double> json_number(const std::string& doc, std::string_view key) {
  const std::string quoted = "\"" + std::string(key) + "\"";
  std::size_t pos = doc.find(quoted);
  if (pos == std::string::npos) return std::nullopt;
  pos = doc.find_first_not_of(" \t\r\n", pos + quoted.size());
  if (pos == std::string::npos || doc[pos] != ':') return std::nullopt;
  const char* begin = doc.c_str() + pos + 1;
  char* end = nullptr;
  const double v = std::strtod(begin, &end);
  if (end == begin) return std::nullopt;
  return v;
}

/// TEMPORARY, until workflow 01 implements ModelConfig::from_json_file: reads
/// just the numeric fields the op shapes need straight out of config.json, so
/// the op benchmarks work while the loader is still a stub. Nothing is
/// hardcoded; once the real parser exists this path is never taken and should
/// be deleted.
ModelConfig scrape_config(const fs::path& file) {
  std::ifstream in(file);
  if (!in) throw std::runtime_error("cannot read " + file.string());
  std::stringstream buf;
  buf << in.rdbuf();
  const std::string doc = buf.str();

  auto require = [&](std::string_view key) {
    const std::optional<double> v = json_number(doc, key);
    if (!v || *v <= 0) {
      throw std::runtime_error(file.string() + ": missing or non-positive \"" + std::string(key) + "\"");
    }
    return *v;
  };
  auto require_int = [&](std::string_view key) { return static_cast<std::int64_t>(require(key)); };

  ModelConfig cfg;
  cfg.vocab_size = require_int("vocab_size");
  cfg.hidden_size = require_int("hidden_size");
  cfg.intermediate_size = require_int("intermediate_size");
  cfg.num_hidden_layers = require_int("num_hidden_layers");
  cfg.num_attention_heads = require_int("num_attention_heads");
  cfg.num_key_value_heads = require_int("num_key_value_heads");
  cfg.max_position_embeddings = require_int("max_position_embeddings");
  cfg.rms_norm_eps = static_cast<float>(require("rms_norm_eps"));
  cfg.rope_theta = static_cast<float>(require("rope_theta"));
  cfg.head_dim = json_number(doc, "head_dim")
                     ? require_int("head_dim")
                     : cfg.hidden_size / cfg.num_attention_heads;
  if (cfg.num_attention_heads % cfg.num_key_value_heads != 0 || cfg.head_dim % 2 != 0) {
    throw std::runtime_error(file.string() + ": unsupported head layout");
  }
  return cfg;
}

ModelConfig load_config(const fs::path& file, std::vector<std::string>& notes) {
  try {
    return ModelConfig::from_json_file(file);
  } catch (const std::logic_error& e) {
    if (!is_not_implemented(e)) throw;
    notes.push_back(std::string(e.what()) + "; read the op shapes straight from config.json instead");
    return scrape_config(file);
  }
}

// ---------------------------------------------------------------------------
// Op benchmarks: ops:: against ref::
// ---------------------------------------------------------------------------

/// The ops.h signatures are identical in both namespaces, so each benchmark is
/// written once, against this table, and run with either implementation.
struct Impl {
  decltype(&ops::embedding) embedding;
  decltype(&ops::matmul) matmul;
  decltype(&ops::rmsnorm) rmsnorm;
  decltype(&ops::silu_mul) silu_mul;
  decltype(&ops::add) add;
  decltype(&ops::softmax) softmax;
  decltype(&ops::rope) rope;
  decltype(&ops::attention) attention;
  decltype(&ops::argmax) argmax;
};

constexpr Impl kOps{&ops::embedding, &ops::matmul, &ops::rmsnorm, &ops::silu_mul, &ops::add,
                    &ops::softmax,   &ops::rope,   &ops::attention, &ops::argmax};
constexpr Impl kRef{&ref::embedding, &ref::matmul, &ref::rmsnorm, &ref::silu_mul, &ref::add,
                    &ref::softmax,   &ref::rope,   &ref::attention, &ref::argmax};

struct OpBench {
  std::string name;   ///< "prefill.matmul.q_proj"; the machine name adds "op." and ".ops"/".ref"
  std::string shape;  ///< for the human table
  std::function<void(const Impl&)> run;
};

/// Uniform in [-1, 1] from a fixed seed. No op has a data-dependent fast
/// path, so the values only have to stay clear of inf, NaN and denormals.
Tensor random_tensor(std::initializer_list<std::int64_t> extents, std::mt19937& rng) {
  Tensor t = Tensor::empty(Shape(extents));
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  float* p = t.f32();
  for (std::int64_t i = 0; i < t.numel(); ++i) p[i] = dist(rng);
  return t;
}

/// One benchmark per (op, shape) that a forward pass actually runs, in
/// forward-pass order, for a prefill of the prompt and for one decode step.
std::vector<OpBench> make_op_benches(const ModelConfig& cfg, const Options& opt) {
  const std::int64_t hidden = cfg.hidden_size;
  const std::int64_t inter = cfg.intermediate_size;
  const std::int64_t vocab = cfg.vocab_size;
  const std::int64_t n_heads = cfg.num_attention_heads;
  const std::int64_t n_kv = cfg.num_key_value_heads;
  const std::int64_t hd = cfg.head_dim;
  const float eps = cfg.rms_norm_eps;
  const float theta = cfg.rope_theta;
  const float scale = cfg.attention_scale();

  std::mt19937 rng(10);
  // Weights are shared by both regimes. The embedding table doubles as the
  // tied LM head, exactly as in the model.
  const Tensor table = random_tensor({vocab, hidden}, rng);
  const Tensor norm_w = random_tensor({hidden}, rng);
  const Tensor w_q = random_tensor({n_heads * hd, hidden}, rng);
  const Tensor w_kv = random_tensor({n_kv * hd, hidden}, rng);
  const Tensor w_o = random_tensor({hidden, n_heads * hd}, rng);
  const Tensor w_up = random_tensor({inter, hidden}, rng);
  const Tensor w_down = random_tensor({hidden, inter}, rng);

  std::vector<OpBench> benches;
  auto push = [&](std::string name, std::string shape, std::function<void(const Impl&)> run) {
    benches.push_back({std::move(name), std::move(shape), std::move(run)});
  };
  auto push_matmul = [&](const std::string& name, std::int64_t rows, const Tensor& w) {
    const Tensor x = random_tensor({rows, w.dim(1)}, rng);
    Tensor y = Tensor::empty(Shape{rows, w.dim(0)});
    push(name, dims({rows, w.dim(1)}) + "x" + dims({w.dim(0), w.dim(1)}) + "^T",
         [x, w, y](const Impl& impl) mutable {
           impl.matmul(x, w, y);
           keep_alive(y.data());
         });
  };

  struct Regime {
    const char* name;
    std::int64_t tokens;     ///< rows in this forward pass
    std::int64_t start_pos;  ///< absolute position of row 0
  };
  const Regime regimes[] = {
      {"prefill", opt.prompt_tokens, 0},
      // The model benchmark's last decode step: the longest context it attends over.
      {"decode", 1, opt.prompt_tokens + opt.decode_tokens - 1},
  };

  for (const Regime& r : regimes) {
    const std::int64_t m = r.tokens;
    const std::int64_t ctx = r.start_pos + m;  // keys the last row attends to
    const std::string pre = std::string(r.name) + ".";

    {
      std::vector<std::int32_t> ids(static_cast<std::size_t>(m));
      std::uniform_int_distribution<std::int32_t> pick(0, static_cast<std::int32_t>(vocab - 1));
      for (std::int32_t& id : ids) id = pick(rng);
      Tensor out = Tensor::empty(Shape{m, hidden});
      push(pre + "embedding", "ids" + dims({m}) + " -> " + dims({m, hidden}),
           [table, ids, out](const Impl& impl) mutable {
             impl.embedding(table, ids, out);
             keep_alive(out.data());
           });
    }
    {
      const Tensor x = random_tensor({m, hidden}, rng);
      Tensor out = Tensor::empty(Shape{m, hidden});
      push(pre + "rmsnorm", dims({m, hidden}), [x, norm_w, eps, out](const Impl& impl) mutable {
        impl.rmsnorm(x, norm_w, eps, out);
        keep_alive(out.data());
      });
    }
    push_matmul(pre + "matmul.q_proj", m, w_q);
    push_matmul(pre + "matmul.kv_proj", m, w_kv);  // k_proj and v_proj: twice per layer
    {
      Tensor q = random_tensor({m, n_heads, hd}, rng);
      Tensor k = random_tensor({m, n_kv, hd}, rng);
      std::vector<std::int64_t> positions(static_cast<std::size_t>(m));
      std::iota(positions.begin(), positions.end(), r.start_pos);
      // In place, so every call rotates the previous call's output. A
      // rotation preserves the norm, so the data never degenerates.
      push(pre + "rope", "q" + dims({m, n_heads, hd}) + " k" + dims({m, n_kv, hd}),
           [q, k, positions, theta](const Impl& impl) mutable {
             impl.rope(q, &k, positions, theta);
             keep_alive(q.data());
             keep_alive(k.data());
           });
    }
    {
      const Tensor q = random_tensor({m, n_heads, hd}, rng);
      const Tensor k = random_tensor({ctx, n_kv, hd}, rng);
      const Tensor v = random_tensor({ctx, n_kv, hd}, rng);
      Tensor out = Tensor::empty(Shape{m, n_heads, hd});
      const std::int64_t start = r.start_pos;
      push(pre + "attention", "q" + dims({m, n_heads, hd}) + " kv" + dims({ctx, n_kv, hd}),
           [q, k, v, start, scale, out](const Impl& impl) mutable {
             impl.attention(q, k, v, start, scale, out);
             keep_alive(out.data());
           });
    }
    {
      // Shaped like the attention scores: one row per (token, head). In place;
      // after the first call the rows are probabilities, which is just as
      // much work to normalize again.
      Tensor x = random_tensor({m * n_heads, ctx}, rng);
      push(pre + "softmax", dims({m * n_heads, ctx}), [x](const Impl& impl) mutable {
        impl.softmax(x);
        keep_alive(x.data());
      });
    }
    push_matmul(pre + "matmul.o_proj", m, w_o);
    {
      const Tensor a = random_tensor({m, hidden}, rng);
      const Tensor b = random_tensor({m, hidden}, rng);
      Tensor out = Tensor::empty(Shape{m, hidden});
      push(pre + "add", dims({m, hidden}), [a, b, out](const Impl& impl) mutable {
        impl.add(a, b, out);
        keep_alive(out.data());
      });
    }
    push_matmul(pre + "matmul.gate_up", m, w_up);  // gate_proj and up_proj: twice per layer
    {
      const Tensor gate = random_tensor({m, inter}, rng);
      const Tensor up = random_tensor({m, inter}, rng);
      Tensor out = Tensor::empty(Shape{m, inter});
      push(pre + "silu_mul", dims({m, inter}), [gate, up, out](const Impl& impl) mutable {
        impl.silu_mul(gate, up, out);
        keep_alive(out.data());
      });
    }
    push_matmul(pre + "matmul.down_proj", m, w_down);
  }

  // Once per forward pass, for the last token only, in prefill and decode alike.
  push_matmul("last_token.matmul.lm_head", 1, table);
  {
    const Tensor logits = random_tensor({vocab}, rng);
    push("last_token.argmax", dims({vocab}), [logits](const Impl& impl) { g_sink = impl.argmax(logits); });
  }
  return benches;
}

/// Returns how many benchmarks matched the filter.
int run_op_benches(const ModelConfig& cfg, const Options& opt, Report& report) {
  std::vector<OpBench> benches = make_op_benches(cfg, opt);
  std::erase_if(benches, [&](const OpBench& b) { return !selected(opt, "op." + b.name); });
  if (benches.empty()) return 0;

  std::printf("\n== ops:: vs ref:: — median time per call, +/- = (max - min) / 2 / median, "
              "speedup = median of ref/ops over interleaved sample pairs ==\n\n");
  std::printf("%-26s %-26s %10s %6s %9s  %10s %6s %9s  %8s %6s\n", "op", "shape", "ops::", "+/-",
              "calls", "ref::", "+/-", "calls", "speedup", "+/-");
  std::fflush(stdout);
  for (const OpBench& b : benches) {
    const Comparison c = time_interleaved([&] { b.run(kOps); }, [&] { b.run(kRef); }, opt);
    std::printf("%-26s %-26s %10s %5.1f%% %9s  %10s %5.1f%% %9s  %7.2fx %5.1f%%\n", b.name.c_str(),
                b.shape.c_str(), human_time(c.a.per_call.median()).c_str(),
                c.a.per_call.spread() * 100.0, c.a.calls().c_str(),
                human_time(c.b.per_call.median()).c_str(), c.b.per_call.spread() * 100.0,
                c.b.calls().c_str(), c.speedup.median(), c.speedup.spread() * 100.0);
    std::fflush(stdout);

    const std::string name = "op." + b.name;
    report.add_time(name + ".ops", c.a);
    report.add_time(name + ".ref", c.b);
    report.add({name + ".speedup", c.speedup.median(), "x", c.speedup.count(), c.speedup.count(),
                c.speedup.min(), c.speedup.max()});
  }
  return static_cast<int>(benches.size());
}

// ---------------------------------------------------------------------------
// Whole-model benchmarks
// ---------------------------------------------------------------------------

std::int32_t greedy(std::span<const float> logits) {
  const Tensor row =
      Tensor::wrap(const_cast<float*>(logits.data()), Shape{static_cast<std::int64_t>(logits.size())});
  return ops::argmax(row);
}

/// Prefill and decode throughput, then the peak RSS they needed. Returns how
/// many benchmarks matched the filter.
int run_model_benches(const Options& opt, Report& report) {
  const bool want_prefill = selected(opt, "model.prefill");
  const bool want_decode = selected(opt, "model.decode");
  if (!want_prefill && !want_decode) return 0;
  const int matched = (want_prefill ? 1 : 0) + (want_decode ? 1 : 0);

  std::optional<Model> model;
  try {
    model.emplace(Model::from_dir(opt.model_dir, opt.prompt_tokens + opt.decode_tokens));
  } catch (const std::logic_error& e) {
    if (!is_not_implemented(e)) throw;
    report.skip("model.prefill, model.decode, memory.peak_rss.model", e.what());
    return matched;
  }

  // Synthetic token ids: the tokenizer is not part of what is being timed,
  // and no kernel's cost depends on which ids it sees.
  const std::int64_t vocab = model->config().vocab_size;
  std::vector<std::int32_t> prompt(static_cast<std::size_t>(opt.prompt_tokens));
  for (std::size_t i = 0; i < prompt.size(); ++i) {
    prompt[i] = static_cast<std::int32_t>((static_cast<std::int64_t>(i) * 7919 + 13) % vocab);
  }

  // One run is a prefill of the prompt from an empty cache, then a greedy
  // decode of decode_tokens tokens, each step timed with its argmax.
  auto run = [&](double& prefill_s, double& decode_s) {
    model->reset();
    const Clock::time_point t0 = Clock::now();
    std::span<const float> logits = model->forward(prompt, 0);
    prefill_s = seconds_since(t0);

    const Clock::time_point t1 = Clock::now();
    for (std::int64_t i = 0; i < opt.decode_tokens; ++i) {
      const std::int32_t next = greedy(logits);
      logits = model->forward(std::span<const std::int32_t>(&next, 1), opt.prompt_tokens + i);
    }
    keep_alive(logits.data());
    decode_s = seconds_since(t1);
  };

  std::printf("\n== whole model — %lld layers, %lld-token synthetic prompt, greedy decode ==\n\n",
              static_cast<long long>(model->config().num_hidden_layers),
              static_cast<long long>(opt.prompt_tokens));
  std::fflush(stdout);

  double prefill_s = 0.0;
  double decode_s = 0.0;
  run(prefill_s, decode_s);  // warm-up
  std::vector<double> prefill_tps;
  std::vector<double> decode_tps;
  for (int s = 0; s < opt.samples; ++s) {
    run(prefill_s, decode_s);
    prefill_tps.push_back(static_cast<double>(opt.prompt_tokens) / prefill_s);
    decode_tps.push_back(static_cast<double>(opt.decode_tokens) / decode_s);
  }
  const Stats prefill(std::move(prefill_tps));
  const Stats decode(std::move(decode_tps));
  const double rss = peak_rss_mib();

  std::printf("%-26s %14s %6s %14s %14s %5s\n", "", "median", "+/-", "min", "max", "runs");
  auto row = [&](const std::string& label, const Stats& st) {
    std::printf("%-26s %8.1f tok/s %5.1f%% %8.1f tok/s %8.1f tok/s %5d\n", label.c_str(), st.median(),
                st.spread() * 100.0, st.min(), st.max(), st.count());
  };
  if (want_prefill) {
    row("prefill (" + std::to_string(opt.prompt_tokens) + " tokens)", prefill);
    report.add({"model.prefill", prefill.median(), "tok/s", opt.samples, prefill.count(), prefill.min(),
                prefill.max()});
  }
  if (want_decode) {
    row("decode (" + std::to_string(opt.decode_tokens) + " tokens)", decode);
    report.add({"model.decode", decode.median(), "tok/s", opt.samples, decode.count(), decode.min(),
                decode.max()});
  }
  std::printf("%-26s %8.1f MiB  (load + prefill + decode)\n", "peak RSS", rss);
  std::fflush(stdout);
  report.add_value("memory.peak_rss.model", rss, "MiB", 1);
  return matched;
}

// ---------------------------------------------------------------------------

int run(const Options& opt) {
  std::vector<std::string> notes;
  const ModelConfig cfg = load_config(opt.model_dir / "config.json", notes);

  std::printf("minillm_bench — %s, %s, %u hardware threads\n",
              kBuildProblem ? kBuildProblem : "release build", kCompiler,
              std::thread::hardware_concurrency());
  std::printf("model   %s: %lld layers, hidden %lld, mlp %lld, heads %lld/%lld x %lld, vocab %lld\n",
              opt.model_dir.string().c_str(), static_cast<long long>(cfg.num_hidden_layers),
              static_cast<long long>(cfg.hidden_size), static_cast<long long>(cfg.intermediate_size),
              static_cast<long long>(cfg.num_attention_heads),
              static_cast<long long>(cfg.num_key_value_heads), static_cast<long long>(cfg.head_dim),
              static_cast<long long>(cfg.vocab_size));
  std::printf("shapes  prefill = %lld tokens; decode = 1 token at context %lld\n",
              static_cast<long long>(opt.prompt_tokens),
              static_cast<long long>(opt.prompt_tokens + opt.decode_tokens));
  std::printf("timing  %d samples per measurement after a warm-up; op samples last >= %.0f ms\n",
              opt.samples, opt.min_sample_s * 1000.0);
  for (const std::string& n : notes) std::printf("note    %s\n", n.c_str());
  if (kBuildProblem) {
    std::printf("\n*** NOT A RELEASE BUILD (%s): these numbers are meaningless. ***\n", kBuildProblem);
  }
  std::fflush(stdout);

  Report report;
  // The model goes first, so that its peak RSS is not inflated by the op
  // benchmarks' buffers.
  int matched = run_model_benches(opt, report);
  matched += run_op_benches(cfg, opt, report);
  if (matched == 0) {
    std::fprintf(stderr, "minillm_bench: --filter '%s' matched no benchmark\n", opt.filter.c_str());
    return kExitError;
  }

  const double rss = peak_rss_mib();
  std::printf("\npeak RSS of the whole run: %.1f MiB\n", rss);
  report.add_value("memory.peak_rss.process", rss, "MiB", 1);

  report.print_summary();
  report.print_machine_lines();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  try {
    if (const std::optional<int> status = parse_options(argc, argv, opt)) return *status;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "minillm_bench: %s\n", e.what());
    return kExitError;
  }

  if (kBuildProblem && !opt.allow_debug_build) {
    std::fprintf(stderr,
                 "minillm_bench: refusing to run: this binary was %s, and its numbers would be "
                 "meaningless.\nBuild the release preset instead:\n"
                 "  cmake --preset release && cmake --build --preset release\n"
                 "  ./build/release/bench/minillm_bench\n"
                 "(--allow-debug-build runs anyway, to smoke-test the benchmark code itself.)\n",
                 kBuildProblem);
    return kExitError;
  }

  std::error_code ec;
  if (!fs::is_regular_file(opt.model_dir / "config.json", ec)) {
    std::fprintf(stderr,
                 "minillm_bench: no model at '%s' (looked for config.json).\n"
                 "Run `python oracle/generate.py` from the repository root to download it, then run\n"
                 "the benchmark from the repository root or pass --model-dir.\n",
                 opt.model_dir.string().c_str());
    return kExitNoModel;
  }

  try {
    return run(opt);
  } catch (const std::exception& e) {
    std::fflush(stdout);
    std::fprintf(stderr, "minillm_bench: %s\n", e.what());
    return kExitError;
  }
}
