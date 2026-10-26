// minillm benchmarks — owned by workflow 10-bench-ci.
//
// Deliberately no benchmarking framework: timing a kernel is a loop and a
// clock, and the dependency list in docs/ARCHITECTURE.md §6 is closed.
//
// What it measures, meaningful on the `release` preset only (the binary
// refuses to run from a sanitizer or unoptimized build):
//   * every op in ops.h at the model's real shapes, ops:: against ref::, so
//     the speedup of each optimization is visible and attributable
//   * ops::matmul on Q8_0 weights against F32, over whole sweeps of the
//     model's linear layers (a decode token, a prefill), with the plain-read
//     ceiling
//   * prefill and decode tokens/s for the whole model
//   * peak resident memory
//   * what one kernel_parallel_for costs the thread pool, hot and cold
// each at one or more kernel thread counts (--threads), so the scaling of the
// threaded kernels is measured the same way as their speedup over ref::.
//
// Each measurement prints a row of a human-readable table as it finishes, and
// the run ends with one machine-readable line per measurement:
//
//   BENCH {"name":"op.prefill.matmul.q_proj.ops","threads":4,"value":812.4,"unit":"us",...}
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
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "minillm/config.h"
#include "minillm/model.h"
#include "minillm/ops.h"
#include "minillm/runtime.h"
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

/// --threads beyond this is a typo, not an experiment.
constexpr std::int64_t kMaxThreads = 1024;

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
  /// Kernel thread counts, resolved by set_num_threads, without duplicates.
  std::vector<int> threads;
  bool allow_debug_build = false;
};

constexpr const char* kUsage = R"(usage: minillm_bench [options]

Benchmarks every op in ops.h (ops:: vs ref::) at the model's real shapes, then
prefill and decode throughput for the whole model, then peak resident memory,
then the thread pool's cost per kernel_parallel_for.
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
  --threads N[,N...]    kernel thread counts; 0 is the engine's default. With
                        several, every measurement is taken at each count,
                        samples interleaved (default: 0)
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
    } else if (arg == "--threads") {
      opt.threads.clear();
      std::string_view list = value();
      for (;;) {
        const std::size_t comma = list.find(',');
        const std::string item(list.substr(0, comma));
        const std::int64_t n = parse_int(arg, item.c_str(), 0);
        if (n > kMaxThreads) {
          throw std::invalid_argument(std::string(arg) + " allows at most " +
                                      std::to_string(kMaxThreads) + " threads, got " + item);
        }
        set_num_threads(static_cast<int>(n));
        if (std::find(opt.threads.begin(), opt.threads.end(), num_threads()) == opt.threads.end()) {
          opt.threads.push_back(num_threads());
        }
        if (comma == std::string_view::npos) break;
        list.remove_prefix(comma + 1);
      }
    } else if (arg == "--allow-debug-build") {
      opt.allow_debug_build = true;
    } else {
      throw std::invalid_argument("unknown option '" + std::string(arg) + "' (see --help)");
    }
  }
  if (opt.threads.empty()) {
    set_num_threads(0);
    opt.threads.push_back(num_threads());
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

/// One of several functions timed against each other: `call` is what is
/// timed, `setup` runs untimed before each of its samples. It selects the
/// kernel thread count, and returns true if that resized the pool.
struct Contender {
  std::function<bool()> setup;
  std::function<void()> call;
};

/// A setup that runs `ops::` at `threads` kernel threads.
std::function<bool()> use_threads(int threads) {
  return [threads] {
    const bool resized = num_threads() != threads;
    set_num_threads(threads);
    return resized;
  };
}

/// Seconds per call of each contender, one value per round.
using Rounds = std::vector<std::vector<double>>;

/// Times the contenders with their samples interleaved, a b c a b c ..., so
/// that drift in the machine's state — clock frequency, thermals, a
/// background process — lands on every contender of a round instead of
/// skewing the comparison. Ratios are then taken within each round (see
/// ratio_stats). Timed back to back instead, two identical functions differ
/// by up to ~20% on a busy machine.
std::vector<Timing> time_round_robin(const std::vector<Contender>& contenders, const Options& opt,
                                     Rounds& rounds) {
  std::vector<std::int64_t> calls;
  for (const Contender& c : contenders) {
    c.setup();
    calls.push_back(calibrate(c.call, opt));
  }
  rounds.assign(contenders.size(), {});
  for (int s = 0; s < opt.samples; ++s) {
    for (std::size_t i = 0; i < contenders.size(); ++i) {
      // A resized pool's threads have not run anything yet.
      if (contenders[i].setup()) contenders[i].call();
      rounds[i].push_back(time_sample(contenders[i].call, calls[i]));
    }
  }
  std::vector<Timing> timings;
  for (std::size_t i = 0; i < contenders.size(); ++i) {
    timings.push_back({Stats(rounds[i]), calls[i]});
  }
  return timings;
}

/// num / den for each round: > 1 means `den` was faster.
Stats ratio_stats(const std::vector<double>& num, const std::vector<double>& den) {
  std::vector<double> ratio;
  for (std::size_t s = 0; s < num.size(); ++s) ratio.push_back(num[s] / den[s]);
  return Stats(std::move(ratio));
}

/// One result, as printed on a machine-readable line.
struct Measurement {
  std::string name;
  int threads = 1;  ///< kernel thread count it was measured at
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
  void add_value(std::string name, int threads, double value, std::string unit,
                 std::int64_t iterations) {
    add({std::move(name), threads, value, std::move(unit), iterations, 0, 0.0, 0.0});
  }

  void add_time(std::string name, int threads, const Timing& t) {
    add({std::move(name), threads, t.per_call.median() * 1e6, "us", t.iterations(),
         t.per_call.count(), t.per_call.min() * 1e6, t.per_call.max() * 1e6});
  }

  void add_ratio(std::string name, int threads, const Stats& r) {
    add({std::move(name), threads, r.median(), "x", r.count(), r.count(), r.min(), r.max()});
  }

  void skip(std::string what, std::string why) {
    std::printf("SKIP %s: %s\n", what.c_str(), why.c_str());
    std::fflush(stdout);
    skips_.push_back(std::move(what) + ": " + std::move(why));
  }

  void print_machine_lines() const {
    for (const Measurement& m : results_) {
      std::string line = "BENCH {\"name\":\"" + m.name + "\",\"threads\":" + std::to_string(m.threads) +
                         strprintf(",\"value\":%.6g", m.value) + ",\"unit\":\"" + m.unit +
                         "\",\"iterations\":" + std::to_string(m.iterations);
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

  // ref:: never uses the kernel threads; it is timed once per op, in the
  // same rounds as ops:: at every thread count.
  const bool scaling = opt.threads.size() > 1;
  std::printf("\n== ops:: vs ref:: — median time per call, +/- = (max - min) / 2 / median, "
              "speedup = median of ref/ops over interleaved samples");
  if (scaling) {
    std::printf(", scaling = the same for ops:: at %d thread(s) / ops:: at thr", opt.threads.front());
  }
  std::printf(" ==\n\n");
  std::printf("%-26s %-26s %3s %10s %6s %9s  %10s %6s %9s  %8s %6s", "op", "shape", "thr", "ops::",
              "+/-", "calls", "ref::", "+/-", "calls", "speedup", "+/-");
  if (scaling) std::printf("  %8s %6s", "scaling", "+/-");
  std::printf("\n");
  std::fflush(stdout);
  for (const OpBench& b : benches) {
    std::vector<Contender> contenders;
    contenders.push_back({[] { return false; }, [&] { b.run(kRef); }});
    for (const int threads : opt.threads) {
      contenders.push_back({use_threads(threads), [&] { b.run(kOps); }});
    }
    Rounds rounds;
    const std::vector<Timing> t = time_round_robin(contenders, opt, rounds);

    const std::string name = "op." + b.name;
    report.add_time(name + ".ref", 1, t[0]);
    for (std::size_t i = 1; i < t.size(); ++i) {
      const int threads = opt.threads[i - 1];
      const Stats speedup = ratio_stats(rounds[0], rounds[i]);
      std::printf("%-26s %-26s %3d %10s %5.1f%% %9s  %10s %5.1f%% %9s  %7.2fx %5.1f%%",
                  b.name.c_str(), b.shape.c_str(), threads,
                  human_time(t[i].per_call.median()).c_str(), t[i].per_call.spread() * 100.0,
                  t[i].calls().c_str(), human_time(t[0].per_call.median()).c_str(),
                  t[0].per_call.spread() * 100.0, t[0].calls().c_str(), speedup.median(),
                  speedup.spread() * 100.0);
      report.add_time(name + ".ops", threads, t[i]);
      report.add_ratio(name + ".speedup", threads, speedup);
      if (scaling && i > 1) {
        const Stats scale = ratio_stats(rounds[1], rounds[i]);
        std::printf("  %7.2fx %5.1f%%", scale.median(), scale.spread() * 100.0);
        report.add_ratio(name + ".scaling", threads, scale);
      }
      std::printf("\n");
    }
    std::fflush(stdout);
  }
  return static_cast<int>(benches.size());
}

// ---------------------------------------------------------------------------
// Q8_0 weights: ops::matmul on Q8_0 against the same call on F32 (workflow 16)
// ---------------------------------------------------------------------------

/// One ops::matmul call with its own buffers, made up front so that a timed
/// sweep is nothing but the calls.
struct MatmulCall {
  Tensor x;
  Tensor w;
  Tensor out;
};

/// The model's linear layers in both formats: every projection of every
/// layer and the lm_head, each with its own weights, F32 and Q8_0 in one
/// allocation each. A decode token reads all of it once, so a sweep must not
/// find any of it in cache from the sweep before: 538 MB of F32 is far beyond
/// the M4's 16 MB of L2, and so is the 143 MB of Q8_0.
struct LinearWeights {
  std::vector<float> f32_storage;
  std::vector<BlockQ8_0> q8_storage;
  std::vector<Tensor> f32;  ///< forward-pass order; the last one is the lm_head
  std::vector<Tensor> q8;   ///< the same shapes
};

LinearWeights make_linear_weights(const ModelConfig& cfg) {
  const std::int64_t hidden = cfg.hidden_size;
  std::vector<Shape> shapes;
  for (std::int64_t l = 0; l < cfg.num_hidden_layers; ++l) {
    for (const Shape& s : {Shape{cfg.q_dim(), hidden}, Shape{cfg.kv_dim(), hidden},
                           Shape{cfg.kv_dim(), hidden}, Shape{hidden, cfg.q_dim()},
                           Shape{cfg.intermediate_size, hidden}, Shape{cfg.intermediate_size, hidden},
                           Shape{hidden, cfg.intermediate_size}}) {
      shapes.push_back(s);
    }
  }
  shapes.push_back(Shape{cfg.vocab_size, hidden});

  std::int64_t numel = 0;
  for (const Shape& s : shapes) numel += s.numel();
  LinearWeights lw;
  lw.f32_storage.resize(static_cast<std::size_t>(numel));
  lw.q8_storage.resize(static_cast<std::size_t>(numel / kQ8_0Block));

  // No kernel's speed depends on the values, so a fast generator does:
  // uniform F32 in [-1, 1), and Q8_0 blocks with q in [-127, 127] and scales
  // in [2^-10, 2^-5), the range the real checkpoint's scales fall in.
  std::uint64_t s = 0x9e3779b97f4a7c15u;
  auto next = [&s] {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  };
  for (float& v : lw.f32_storage) {
    v = static_cast<float>(static_cast<std::int32_t>(next() >> 32)) * 0x1p-31f;
  }
  for (BlockQ8_0& b : lw.q8_storage) {
    b.d = static_cast<std::uint16_t>(((5 + next() % 5) << 10) | (next() & 0x3ffu));
    for (std::int8_t& q : b.qs) q = static_cast<std::int8_t>(static_cast<int>(next() % 255) - 127);
  }

  float* f = lw.f32_storage.data();
  BlockQ8_0* q = lw.q8_storage.data();
  for (const Shape& shape : shapes) {
    lw.f32.push_back(Tensor::wrap(f, shape));
    lw.q8.push_back(Tensor::wrap(q, shape, DType::Q8_0));
    f += shape.numel();
    q += shape.numel() / kQ8_0Block;
  }
  return lw;
}

/// ops::matmul(x, w) with an m-row x for each weight in `weights`.
std::vector<MatmulCall> make_matmul_calls(const std::vector<Tensor>& weights, std::int64_t m,
                                          std::mt19937& rng) {
  std::vector<MatmulCall> calls;
  for (const Tensor& w : weights) {
    calls.push_back({random_tensor({m, w.dim(1)}, rng), w, Tensor::empty(Shape{m, w.dim(0)})});
  }
  return calls;
}

void sweep(std::vector<MatmulCall>& calls) {
  for (MatmulCall& c : calls) {
    ops::matmul(c.x, c.w, c.out);
    keep_alive(c.out.data());
  }
}

/// Sums n floats, split across the kernel threads: a plain sequential read,
/// the ceiling any weight-streaming kernel can reach on this machine.
void plain_read(const float* p, std::int64_t n) {
  typedef float f32x4 __attribute__((vector_size(16)));
  constexpr std::int64_t kChunk = std::int64_t{1} << 20;  // floats per unit, 4 MB
  const std::int64_t units = (n + kChunk - 1) / kChunk;
  std::vector<float> sums(static_cast<std::size_t>(units));
  kernel_parallel_for(units, 1, [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t u = begin; u < end; ++u) {
      const std::int64_t lo = u * kChunk;
      const std::int64_t hi = std::min(n, lo + kChunk);
      f32x4 acc[4] = {};
      std::int64_t i = lo;
      for (; i + 16 <= hi; i += 16) {
        for (int a = 0; a < 4; ++a) {
          f32x4 v;
          std::memcpy(&v, p + i + 4 * a, sizeof(v));
          acc[a] += v;
        }
      }
      const f32x4 t = (acc[0] + acc[1]) + (acc[2] + acc[3]);
      float sum = (t[0] + t[1]) + (t[2] + t[3]);
      for (; i < hi; ++i) sum += p[i];
      sums[static_cast<std::size_t>(u)] = sum;
    }
  });
  keep_alive(sums.data());
}

double gbps(double bytes, double seconds) { return bytes / seconds / 1e9; }

/// Q8_0 against F32 ops::matmul over whole sweeps of the model's linear
/// layers, at every thread count, with all contenders' samples interleaved:
///
///   q8_0.decode.token    one decode token: every projection plus the lm_head,
///                        m = 1, 211 calls. Memory-bound for F32, so also the
///                        weight bytes per second of each and the plain-read
///                        ceiling over the F32 weights, timed in the same rounds.
///   q8_0.prefill.m64     every projection at m = 64 rows, 210 calls
///   q8_0.prefill.m79     the same at m = 79, the longest oracle prompt
///
/// The headline is speedup = F32 time / Q8_0 time, a median over rounds.
/// Returns how many benchmarks matched the filter.
int run_q8_0_benches(const ModelConfig& cfg, const Options& opt, Report& report) {
  const bool want_decode = selected(opt, "q8_0.decode.token");
  const std::vector<std::int64_t> prefill_rows = {64, 79};
  std::vector<std::int64_t> want_prefill;
  for (const std::int64_t m : prefill_rows) {
    if (selected(opt, "q8_0.prefill.m" + std::to_string(m))) want_prefill.push_back(m);
  }
  if (!want_decode && want_prefill.empty()) return 0;

  const LinearWeights lw = make_linear_weights(cfg);
  const double f32_bytes = static_cast<double>(lw.f32_storage.size() * sizeof(float));
  const double q8_bytes = static_cast<double>(lw.q8_storage.size() * sizeof(BlockQ8_0));
  std::mt19937 rng(16);

  std::printf("\n== Q8_0 vs F32 ops::matmul over whole sweeps of the model's linear layers — "
              "median time per sweep, speedup = median of f32/q8_0 over interleaved samples ==\n\n");
  std::printf("%-30s %3s %10s %6s %10s %6s %8s %6s %9s %9s %9s\n", "sweep", "thr", "f32", "+/-",
              "q8_0", "+/-", "speedup", "+/-", "f32 GB/s", "q8 GB/s", "read GB/s");
  std::fflush(stdout);

  // Rounds of one sweep: each thread count contributes `per_count`
  // contenders, F32 then Q8_0, then for decode the plain read.
  auto measure = [&](const std::string& name, const std::string& label,
                     std::vector<MatmulCall>& f32_calls, std::vector<MatmulCall>& q8_calls,
                     bool bandwidth) {
    std::vector<Contender> contenders;
    for (const int threads : opt.threads) {
      contenders.push_back({use_threads(threads), [&] { sweep(f32_calls); }});
      contenders.push_back({use_threads(threads), [&] { sweep(q8_calls); }});
      if (bandwidth) {
        contenders.push_back({use_threads(threads), [&] {
                                plain_read(lw.f32_storage.data(),
                                           static_cast<std::int64_t>(lw.f32_storage.size()));
                              }});
      }
    }
    Rounds rounds;
    const std::vector<Timing> t = time_round_robin(contenders, opt, rounds);
    const std::size_t per_count = bandwidth ? 3 : 2;
    for (std::size_t c = 0; c < opt.threads.size(); ++c) {
      const int threads = opt.threads[c];
      const std::size_t i = c * per_count;
      const Stats speedup = ratio_stats(rounds[i], rounds[i + 1]);
      const double f32_s = t[i].per_call.median();
      const double q8_s = t[i + 1].per_call.median();
      std::printf("%-30s %3d %10s %5.1f%% %10s %5.1f%% %7.2fx %5.1f%%", label.c_str(), threads,
                  human_time(f32_s).c_str(), t[i].per_call.spread() * 100.0,
                  human_time(q8_s).c_str(), t[i + 1].per_call.spread() * 100.0, speedup.median(),
                  speedup.spread() * 100.0);
      report.add_time(name + ".f32", threads, t[i]);
      report.add_time(name + ".q8_0", threads, t[i + 1]);
      report.add_ratio(name + ".speedup", threads, speedup);
      if (bandwidth) {
        const double read_s = t[i + 2].per_call.median();
        std::printf(" %9.1f %9.1f %9.1f", gbps(f32_bytes, f32_s), gbps(q8_bytes, q8_s),
                    gbps(f32_bytes, read_s));
        report.add_value(name + ".f32.gbps", threads, gbps(f32_bytes, f32_s), "GB/s",
                         t[i].iterations());
        report.add_value(name + ".q8_0.gbps", threads, gbps(q8_bytes, q8_s), "GB/s",
                         t[i + 1].iterations());
        report.add_value("q8_0.read.gbps", threads, gbps(f32_bytes, read_s), "GB/s",
                         t[i + 2].iterations());
      }
      std::printf("\n");
    }
    std::fflush(stdout);
  };

  if (want_decode) {
    std::vector<MatmulCall> f32_calls = make_matmul_calls(lw.f32, 1, rng);
    std::vector<MatmulCall> q8_calls = make_matmul_calls(lw.q8, 1, rng);
    measure("q8_0.decode.token", "decode token (" + std::to_string(f32_calls.size()) + " matmuls)",
            f32_calls, q8_calls, true);
  }
  // Prefill runs the projections only: the lm_head sees the last row alone.
  const std::vector<Tensor> f32_proj(lw.f32.begin(), lw.f32.end() - 1);
  const std::vector<Tensor> q8_proj(lw.q8.begin(), lw.q8.end() - 1);
  for (const std::int64_t m : want_prefill) {
    std::vector<MatmulCall> f32_calls = make_matmul_calls(f32_proj, m, rng);
    std::vector<MatmulCall> q8_calls = make_matmul_calls(q8_proj, m, rng);
    measure("q8_0.prefill.m" + std::to_string(m),
            "prefill m=" + std::to_string(m) + " (" + std::to_string(f32_calls.size()) + " matmuls)",
            f32_calls, q8_calls, false);
  }
  return (want_decode ? 1 : 0) + static_cast<int>(want_prefill.size());
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

  // With several thread counts, the runs are interleaved across them, one
  // run at each count per round, as the op samples are.
  const std::size_t n_counts = opt.threads.size();
  double prefill_s = 0.0;
  double decode_s = 0.0;
  for (const int threads : opt.threads) {
    set_num_threads(threads);
    run(prefill_s, decode_s);  // warm-up
  }
  std::vector<std::vector<double>> prefill_tps(n_counts);
  std::vector<std::vector<double>> decode_tps(n_counts);
  for (int s = 0; s < opt.samples; ++s) {
    for (std::size_t i = 0; i < n_counts; ++i) {
      set_num_threads(opt.threads[i]);
      run(prefill_s, decode_s);
      prefill_tps[i].push_back(static_cast<double>(opt.prompt_tokens) / prefill_s);
      decode_tps[i].push_back(static_cast<double>(opt.decode_tokens) / decode_s);
    }
  }
  const double rss = peak_rss_mib();

  std::printf("%-26s %3s %14s %6s %14s %14s %5s\n", "", "thr", "median", "+/-", "min", "max", "runs");
  auto row = [&](const std::string& label, int threads, const Stats& st) {
    std::printf("%-26s %3d %8.1f tok/s %5.1f%% %8.1f tok/s %8.1f tok/s %5d\n", label.c_str(), threads,
                st.median(), st.spread() * 100.0, st.min(), st.max(), st.count());
  };
  for (std::size_t i = 0; i < n_counts; ++i) {
    const int threads = opt.threads[i];
    const Stats prefill(prefill_tps[i]);
    const Stats decode(decode_tps[i]);
    if (want_prefill) {
      row("prefill (" + std::to_string(opt.prompt_tokens) + " tokens)", threads, prefill);
      report.add({"model.prefill", threads, prefill.median(), "tok/s", opt.samples, prefill.count(),
                  prefill.min(), prefill.max()});
    }
    if (want_decode) {
      row("decode (" + std::to_string(opt.decode_tokens) + " tokens)", threads, decode);
      report.add({"model.decode", threads, decode.median(), "tok/s", opt.samples, decode.count(),
                  decode.min(), decode.max()});
    }
  }
  // Peak RSS covers the runs at every count; it is filed under the largest.
  const int most = *std::max_element(opt.threads.begin(), opt.threads.end());
  std::printf("%-26s %8.1f MiB  (load + prefill + decode)\n", "peak RSS", rss);
  std::fflush(stdout);
  report.add_value("memory.peak_rss.model", most, rss, "MiB", 1);
  return matched;
}

// ---------------------------------------------------------------------------
// Runtime benchmarks: what one kernel_parallel_for costs
// ---------------------------------------------------------------------------

/// The pool's own cost per kernel_parallel_for, at each thread count: handing
/// out the job, waking workers, and waiting for the last chunk. The body does
/// next to nothing, so that is all there is to time. `hot` calls back to back,
/// as a forward pass does about 240 times per decode step; `cold` calls once
/// the workers have gone to sleep, as the first call after a pause does.
/// Returns how many benchmarks matched the filter.
int run_runtime_benches(const Options& opt, Report& report) {
  const bool want_hot = selected(opt, "runtime.parallel_for.hot");
  const bool want_cold = selected(opt, "runtime.parallel_for.cold");
  if (!want_hot && !want_cold) return 0;

  // Enough items for several chunks on every thread.
  constexpr std::int64_t kItems = 1024;
  // Far longer than a worker spins before it sleeps.
  constexpr auto kPause = std::chrono::milliseconds(2);
  constexpr int kColdCalls = 20;  // per sample
  std::vector<std::int64_t> sink(static_cast<std::size_t>(kItems));
  const auto body = [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t i = begin; i < end; ++i) ++sink[static_cast<std::size_t>(i)];
  };
  const auto call = [&] {
    kernel_parallel_for(kItems, 1, std::ref(body));
    keep_alive(sink.data());
  };

  std::printf("\n== runtime — one kernel_parallel_for over %lld items with a trivial body ==\n\n",
              static_cast<long long>(kItems));
  std::printf("%-26s %3s %10s %6s %9s\n", "", "thr", "median", "+/-", "calls");
  std::fflush(stdout);
  const auto row = [&](const std::string& name, int threads, const Timing& t) {
    std::printf("%-26s %3d %10s %5.1f%% %9s\n", name.c_str(), threads,
                human_time(t.per_call.median()).c_str(), t.per_call.spread() * 100.0,
                t.calls().c_str());
    std::fflush(stdout);
    report.add_time(name, threads, t);
  };

  int matched = 0;
  if (want_hot) {
    ++matched;
    std::vector<Contender> contenders;
    for (const int threads : opt.threads) contenders.push_back({use_threads(threads), call});
    Rounds rounds;
    const std::vector<Timing> t = time_round_robin(contenders, opt, rounds);
    for (std::size_t i = 0; i < t.size(); ++i) {
      row("runtime.parallel_for.hot", opt.threads[i], t[i]);
    }
  }
  if (want_cold) {
    ++matched;
    // Each sample is the median of kColdCalls single calls, each made after
    // kPause of nothing; samples are interleaved across thread counts.
    std::vector<std::vector<double>> samples(opt.threads.size());
    for (int s = 0; s < opt.samples; ++s) {
      for (std::size_t i = 0; i < opt.threads.size(); ++i) {
        set_num_threads(opt.threads[i]);
        call();  // a resized pool's threads have not run anything yet
        std::vector<double> calls;
        for (int c = 0; c < kColdCalls; ++c) {
          std::this_thread::sleep_for(kPause);
          const Clock::time_point start = Clock::now();
          call();
          calls.push_back(seconds_since(start));
        }
        samples[i].push_back(Stats(std::move(calls)).median());
      }
    }
    for (std::size_t i = 0; i < opt.threads.size(); ++i) {
      row("runtime.parallel_for.cold", opt.threads[i], {Stats(samples[i]), kColdCalls});
    }
  }
  return matched;
}

// ---------------------------------------------------------------------------

int run(const Options& opt) {
  const ModelConfig cfg = ModelConfig::from_json_file(opt.model_dir / "config.json");

  std::printf("minillm_bench — %s, %s, %u hardware threads\n",
              kBuildProblem ? kBuildProblem : "release build", kCompiler,
              std::thread::hardware_concurrency());
  std::string counts;
  for (const int threads : opt.threads) {
    counts += (counts.empty() ? "" : ", ") + std::to_string(threads);
  }
  std::printf("threads kernels run on %s thread(s) (--threads)\n", counts.c_str());
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
  if (kBuildProblem) {
    std::printf("\n*** NOT A RELEASE BUILD (%s): these numbers are meaningless. ***\n", kBuildProblem);
  }
  std::fflush(stdout);

  Report report;
  // The model goes first, so that its peak RSS is not inflated by the op
  // benchmarks' buffers.
  int matched = run_model_benches(opt, report);
  matched += run_op_benches(cfg, opt, report);
  matched += run_q8_0_benches(cfg, opt, report);
  matched += run_runtime_benches(opt, report);
  if (matched == 0) {
    std::fprintf(stderr, "minillm_bench: --filter '%s' matched no benchmark\n", opt.filter.c_str());
    return kExitError;
  }

  const double rss = peak_rss_mib();
  std::printf("\npeak RSS of the whole run: %.1f MiB\n", rss);
  const int most = *std::max_element(opt.threads.begin(), opt.threads.end());
  report.add_value("memory.peak_rss.process", most, rss, "MiB", 1);

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
