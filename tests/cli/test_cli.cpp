// Workflow 09-cli: the `minillm` executable.
//
// Three layers, cheapest first:
//
//   1. The pieces main() is assembled from, called in-process through the
//      private headers in src/cli/: argument parsing, the model-directory
//      check, the UTF-8 stream buffer, and the generation loop driven by a
//      fake model.
//   2. The real binary, spawned as a child process: exit statuses and
//      messages, and that no exception escapes main().
//   3. End to end: the real binary on the real checkpoint, greedy, against
//      oracle/data/model/p{i}_greedy_ids.npy id for id, and its streamed
//      stdout against a one-call decode of the same ids.
//
// This target links the tokenizer and the sampler but not the model (see
// tests/CMakeLists.txt), so the forward pass is only ever reached through the
// binary. The binary is found relative to this test's own executable:
// <build>/tests/test_cli -> <build>/src/minillm.

#include <doctest/doctest.h>

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <regex>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../../src/cli/args.h"
#include "../../src/cli/generate.h"
#include "../../src/cli/utf8_stream.h"
#include "common/allclose.h"
#include "common/oracle.h"
#include "minillm/tokenizer.h"

extern char** environ;

namespace fs = std::filesystem;
using namespace minillm;
using namespace minillm::test;

namespace {

// ===========================================================================
// Helpers
// ===========================================================================

bool contains(std::string_view haystack, std::string_view needle) {
  return haystack.find(needle) != std::string_view::npos;
}

void skip(const std::string& reason) {
  note_skip(reason);
  MESSAGE("SKIP: " << reason);
}

/// True for the std::logic_error a stub module throws.
bool is_stub(const std::exception& e) {
  return std::string_view(e.what()).starts_with("not implemented");
}

cli::Options parse(std::initializer_list<std::string_view> args) {
  const std::vector<std::string_view> v(args);
  return cli::parse_args(v);
}

/// The UsageError message parse() throws, or "" if the arguments parse.
std::string parse_error(std::initializer_list<std::string_view> args) {
  try {
    parse(args);
  } catch (const cli::UsageError& e) {
    return e.what();
  }
  return {};
}

/// The runtime_error message check_model_dir() throws, or "" if it passes.
std::string model_dir_error(const fs::path& dir) {
  try {
    cli::check_model_dir(dir);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return {};
}

/// Strict UTF-8 validation: no overlong forms, surrogates, or code points
/// past U+10FFFF. Deliberately stricter than Utf8Stream, which only needs to
/// know where a character ends.
bool is_valid_utf8(std::string_view s) {
  std::size_t i = 0;
  while (i < s.size()) {
    const auto c = static_cast<unsigned char>(s[i]);
    std::size_t len = 0;
    unsigned char lo = 0x80;  // allowed range of the second byte
    unsigned char hi = 0xBF;
    if (c < 0x80) {
      len = 1;
    } else if (c >= 0xC2 && c <= 0xDF) {
      len = 2;
    } else if (c == 0xE0) {
      len = 3;
      lo = 0xA0;
    } else if (c == 0xED) {
      len = 3;
      hi = 0x9F;
    } else if (c >= 0xE1 && c <= 0xEF) {
      len = 3;
    } else if (c == 0xF0) {
      len = 4;
      lo = 0x90;
    } else if (c == 0xF4) {
      len = 4;
      hi = 0x8F;
    } else if (c >= 0xF1 && c <= 0xF3) {
      len = 4;
    } else {
      return false;
    }
    if (i + len > s.size()) return false;
    for (std::size_t k = 1; k < len; ++k) {
      const auto b = static_cast<unsigned char>(s[i + k]);
      if (k == 1 ? (b < lo || b > hi) : (b & 0xC0) != 0x80) return false;
    }
    i += len;
  }
  return true;
}

std::string ids_to_string(std::span<const std::int32_t> ids) {
  std::string s;
  for (const std::int32_t id : ids) {
    if (!s.empty()) s += ' ';
    s += std::to_string(id);
  }
  return s;
}

std::string read_file(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_file(const fs::path& p, std::string_view content = {}) {
  std::ofstream(p, std::ios::binary) << content;
}

/// Per-process scratch directory, removed at exit.
const fs::path& scratch_dir() {
  struct Dir {
    fs::path path =
        fs::temp_directory_path() / ("minillm_test_cli_" + std::to_string(::getpid()));
    Dir() { fs::create_directories(path); }
    ~Dir() {
      std::error_code ec;
      fs::remove_all(path, ec);
    }
  };
  static const Dir dir;
  return dir.path;
}

/// A fresh, empty directory under scratch_dir().
fs::path make_temp_dir(std::string_view name) {
  const fs::path p = scratch_dir() / name;
  fs::remove_all(p);
  fs::create_directories(p);
  return p;
}

/// The checkpoint directory oracle/generate.py downloads into.
const fs::path& checkpoint_dir() {
  static const fs::path dir = models_dir() / "SmolLM2-135M";
  return dir;
}

/// The real tokenizer, or nullopt after recording why it is unavailable.
std::optional<Tokenizer> load_tokenizer() {
  const fs::path file = checkpoint_dir() / "tokenizer.json";
  std::error_code ec;
  if (!fs::is_regular_file(file, ec)) {
    skip("cli: missing " + file.string() + " — run `python oracle/generate.py`");
    return std::nullopt;
  }
  try {
    return Tokenizer::from_dir(checkpoint_dir());
  } catch (const std::logic_error& e) {
    if (!is_stub(e)) throw;
    skip(std::string("cli: the tokenizer is still a stub (") + e.what() + ")");
    return std::nullopt;
  }
}

/// "eos_token_id" from the checkpoint's config.json: the id the CLI stops on.
/// A regex rather than a JSON parser, because this target links none.
std::int32_t config_eos_id() {
  const std::string json = read_file(checkpoint_dir() / "config.json");
  static const std::regex re(R"re("eos_token_id"\s*:\s*(\d+))re");
  std::smatch m;
  REQUIRE_MESSAGE(std::regex_search(json, m, re), "no integer eos_token_id in config.json");
  return static_cast<std::int32_t>(std::stol(m[1].str()));
}

// ---------------------------------------------------------------------------
// Child processes
// ---------------------------------------------------------------------------

struct RunResult {
  int exit_code = -1;     ///< exit status, or 128 + signal number if killed
  bool signaled = false;  ///< killed by a signal, e.g. SIGABRT from std::terminate
  std::string out;
  std::string err;
};

/// A started child. Its stdout and stderr go to files, not pipes, so any
/// number of children can run at once without one stalling on a full pipe.
struct Child {
  pid_t pid = -1;
  fs::path out_path;
  fs::path err_path;
};

Child spawn(const fs::path& exe, const std::vector<std::string>& args) {
  static int counter = 0;
  Child child;
  const std::string tag = std::to_string(counter++);
  child.out_path = scratch_dir() / (tag + ".out");
  child.err_path = scratch_dir() / (tag + ".err");

  std::vector<std::string> storage{exe.string()};
  storage.insert(storage.end(), args.begin(), args.end());
  std::vector<char*> argv;
  for (std::string& a : storage) argv.push_back(a.data());
  argv.push_back(nullptr);

  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
  posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, child.out_path.c_str(),
                                   O_WRONLY | O_CREAT | O_TRUNC, 0644);
  posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, child.err_path.c_str(),
                                   O_WRONLY | O_CREAT | O_TRUNC, 0644);
  const int rc = posix_spawn(&child.pid, argv[0], &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  if (rc != 0) {
    throw std::runtime_error("posix_spawn " + exe.string() + ": " + std::strerror(rc));
  }
  return child;
}

RunResult wait_for(const Child& child) {
  int status = 0;
  while (::waitpid(child.pid, &status, 0) < 0) {
    if (errno != EINTR) throw std::runtime_error(std::string("waitpid: ") + std::strerror(errno));
  }
  RunResult r;
  if (WIFEXITED(status)) {
    r.exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    r.signaled = true;
    r.exit_code = 128 + WTERMSIG(status);
  }
  r.out = read_file(child.out_path);
  r.err = read_file(child.err_path);
  return r;
}

fs::path self_exe() {
#if defined(__APPLE__)
  std::uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  std::string buf(size, '\0');
  if (_NSGetExecutablePath(buf.data(), &size) != 0) return {};
  buf.resize(std::strlen(buf.c_str()));
  return fs::weakly_canonical(buf);
#else
  std::error_code ec;
  return fs::read_symlink("/proc/self/exe", ec);
#endif
}

const fs::path& cli_exe() {
  static const fs::path exe = self_exe().parent_path().parent_path() / "src" / "minillm";
  return exe;
}

/// True if the binary is there to run; otherwise records a skip.
bool have_cli() {
  std::error_code ec;
  if (fs::is_regular_file(cli_exe(), ec)) return true;
  skip("cli: minillm binary not found at " + cli_exe().string() +
       " — build the minillm_cli target");
  return false;
}

RunResult run_cli(const std::vector<std::string>& args) { return wait_for(spawn(cli_exe(), args)); }

/// The ids on the stderr line "<label>: 1 2 3" that --print-ids writes.
std::optional<std::vector<std::int32_t>> printed_ids(const std::string& err,
                                                     std::string_view label) {
  const std::string prefix = std::string(label) + ":";
  std::istringstream lines(err);
  for (std::string line; std::getline(lines, line);) {
    if (!line.starts_with(prefix)) continue;
    std::istringstream in(line.substr(prefix.size()));
    std::vector<std::int32_t> ids;
    for (std::int32_t id = 0; in >> id;) ids.push_back(id);
    return ids;
  }
  return std::nullopt;
}

std::string line_containing(const std::string& text, std::string_view needle) {
  std::istringstream lines(text);
  for (std::string line; std::getline(lines, line);) {
    if (contains(line, needle)) return line;
  }
  return {};
}

// ---------------------------------------------------------------------------
// A fake model and sampler for the generation loop
// ---------------------------------------------------------------------------

constexpr std::size_t kFakeVocab = 16;

/// Answers forward() call i with one-hot logits on script[i], tracks the
/// cache length the way Model does, and logs every call.
struct FakeModel {
  struct Cache {
    std::int64_t len = 0;
    std::int64_t seq_len() const { return len; }
  };

  std::vector<std::int32_t> script;
  std::vector<std::string>* log = nullptr;
  std::size_t calls = 0;
  Cache cache;
  std::vector<float> logits = std::vector<float>(kFakeVocab, 0.0f);

  std::span<const float> forward(std::span<const std::int32_t> tokens, std::int64_t start_pos) {
    REQUIRE_MESSAGE(calls < script.size(), "forward() called more often than scripted");
    log->push_back("forward [" + ids_to_string(tokens) + "] at " + std::to_string(start_pos));
    cache.len = start_pos + static_cast<std::int64_t>(tokens.size());
    std::fill(logits.begin(), logits.end(), 0.0f);
    logits[static_cast<std::size_t>(script[calls++])] = 1.0f;
    return logits;
  }
  Cache& kv_cache() { return cache; }
};

/// Greedy, and remembers the history it was shown each time.
struct FakeSampler {
  std::vector<std::string> histories;

  std::int32_t sample(std::span<float> logits, std::span<const std::int32_t> history) {
    histories.push_back(ids_to_string(history));
    return static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) -
                                     logits.begin());
  }
};

struct LoopRun {
  cli::Generation gen;
  std::string log;  ///< forward calls and on_token calls, in order, "; "-separated
  std::vector<std::string> histories;
};

LoopRun run_loop(std::vector<std::int32_t> script, std::int64_t max_tokens, std::int32_t eos_id,
                 bool penalize_history = false) {
  std::vector<std::string> log;
  FakeModel model;
  model.script = std::move(script);
  model.log = &log;
  FakeSampler sampler;
  const std::vector<std::int32_t> prompt = {5, 6, 7};

  LoopRun run;
  run.gen = cli::generate(model, sampler, prompt, max_tokens, eos_id, penalize_history,
                          [&](std::int32_t id) { log.push_back("token " + std::to_string(id)); });
  for (const std::string& entry : log) run.log += (run.log.empty() ? "" : "; ") + entry;
  run.histories = sampler.histories;
  return run;
}

// ---------------------------------------------------------------------------
// End-to-end runs
// ---------------------------------------------------------------------------

// The prompts oracle/generate.py feeds the model (its PROMPTS list). The CLI
// prints the ids it tokenizes them to and the test compares those against
// p{i}_input_ids.npy, so these cannot drift from the oracle unnoticed.
constexpr std::array<std::string_view, 3> kOraclePrompts = {
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a village at the edge of the forest,",
};

// Not an oracle prompt. None of the three above continues with a non-ASCII
// character, so this one is here to stream multi-byte output through the real
// binary: transformers continues it greedily with
// ")\n\nThe city of Tokyo (東京) is located in", and byte-level BPE splits
// both CJK characters across tokens.
constexpr std::string_view kMultiBytePrompt = "Tokyo (東京";
constexpr std::int64_t kMultiByteTokens = 16;

struct EndToEnd {
  std::string skip_reason;          ///< non-empty: nothing ran
  std::array<RunResult, 3> oracle;  ///< kOraclePrompts, as many tokens as the oracle, --ignore-eos
  RunResult stop_at_eos;            ///< kOraclePrompts[1] again, stopping at eos
  RunResult multibyte;              ///< kMultiBytePrompt
};

std::vector<std::string> greedy_args(std::string_view prompt, std::size_t max_tokens,
                                     bool ignore_eos) {
  std::vector<std::string> args = {"--model",       checkpoint_dir().string(),
                                   "--prompt",      std::string(prompt),
                                   "--max-tokens",  std::to_string(max_tokens),
                                   "--temperature", "0",
                                   "--print-ids"};
  if (ignore_eos) args.emplace_back("--ignore-eos");
  return args;
}

/// Runs every end-to-end invocation the first time a test asks for one, all
/// at once in parallel: each is a full model load plus a few dozen decode
/// steps, and they are independent.
const EndToEnd& end_to_end() {
  static const EndToEnd e2e = [] {
    EndToEnd r;
    std::error_code ec;
    if (!fs::is_regular_file(cli_exe(), ec)) {
      r.skip_reason = "minillm binary not found at " + cli_exe().string() +
                      " — build the minillm_cli target";
      return r;
    }
    std::string missing;
    for (std::size_t i = 0; i < kOraclePrompts.size(); ++i) {
      for (const char* kind : {"input_ids", "greedy_ids"}) {
        const std::string rel = "model/p" + std::to_string(i) + "_" + kind + ".npy";
        if (oracle_has(rel)) continue;
        if (!missing.empty()) missing += ", ";
        missing += rel;
      }
    }
    if (!missing.empty()) {
      r.skip_reason = "missing oracle data: " + missing + " — run `python oracle/generate.py`";
      return r;
    }
    if (const std::string err = model_dir_error(checkpoint_dir()); !err.empty()) {
      r.skip_reason = err + " — run `python oracle/generate.py`";
      return r;
    }

    std::array<std::size_t, kOraclePrompts.size()> n_oracle{};
    for (std::size_t i = 0; i < kOraclePrompts.size(); ++i) {
      n_oracle[i] =
          oracle_npy("model/p" + std::to_string(i) + "_greedy_ids.npy").as_i32_narrowing().size();
    }

    // The oracle never stops early, so --ignore-eos to match it token for token.
    std::vector<Child> children;
    for (std::size_t i = 0; i < kOraclePrompts.size(); ++i) {
      children.push_back(spawn(cli_exe(), greedy_args(kOraclePrompts[i], n_oracle[i], true)));
    }
    children.push_back(spawn(cli_exe(), greedy_args(kOraclePrompts[1], n_oracle[1], false)));
    children.push_back(spawn(cli_exe(), greedy_args(kMultiBytePrompt, kMultiByteTokens, false)));
    for (std::size_t i = 0; i < kOraclePrompts.size(); ++i) r.oracle[i] = wait_for(children[i]);
    r.stop_at_eos = wait_for(children[3]);
    r.multibyte = wait_for(children[4]);

    // Until every module the binary uses is implemented, it stops at the
    // first stub it reaches and says so.
    for (const RunResult* run : {&r.oracle[0], &r.oracle[1], &r.oracle[2], &r.stop_at_eos,
                                 &r.multibyte}) {
      if (run->exit_code != 0 && contains(run->err, "not implemented")) {
        r.skip_reason =
            "the engine is incomplete — " + line_containing(run->err, "not implemented");
        break;
      }
    }
    return r;
  }();
  return e2e;
}

/// The end-to-end results, or nullptr after recording why there are none.
const EndToEnd* e2e_or_skip() {
  const EndToEnd& e2e = end_to_end();
  if (e2e.skip_reason.empty()) return &e2e;
  skip("cli end to end: " + e2e.skip_reason);
  return nullptr;
}

/// Checks that a run succeeded and returns the ids it generated.
std::vector<std::int32_t> generated_ids(const RunResult& run) {
  CHECK_MESSAGE(run.exit_code == 0, "minillm exited with " << run.exit_code << ":\n" << run.err);
  const std::optional<std::vector<std::int32_t>> ids = printed_ids(run.err, "generated ids");
  CHECK_MESSAGE(ids.has_value(), "no 'generated ids:' line on stderr:\n" << run.err);
  return ids.value_or(std::vector<std::int32_t>{});
}

// Every flag the CLI documents, and whether it takes a value.
struct FlagSpec {
  std::string_view name;
  bool takes_value;
};
constexpr std::array<FlagSpec, 12> kFlags = {{
    {"--model", true},
    {"--prompt", true},
    {"--max-tokens", true},
    {"--temperature", true},
    {"--top-k", true},
    {"--top-p", true},
    {"--repetition-penalty", true},
    {"--seed", true},
    {"--threads", true},
    {"--ignore-eos", false},
    {"--print-ids", false},
    {"--help", false},
}};

}  // namespace

// ===========================================================================
// 1a. Argument parsing
// ===========================================================================

TEST_CASE("parse_args: defaults are the documented ones") {
  const cli::Options o = parse({"--model", "models/x", "--prompt", "Hello"});
  CHECK(o.model_dir == fs::path("models/x"));
  CHECK(o.prompt == "Hello");
  CHECK(o.max_tokens == 64);
  CHECK(o.sampling.temperature == 0.0f);
  CHECK(o.sampling.top_k == 0);
  CHECK(o.sampling.top_p == 1.0f);
  CHECK(o.sampling.repetition_penalty == 1.0f);
  CHECK(o.sampling.seed == 42u);
  CHECK(o.threads == 0);
  CHECK_FALSE(o.ignore_eos);
  CHECK_FALSE(o.print_ids);
  CHECK_FALSE(o.help);
}

TEST_CASE("parse_args: every flag round-trips, as two arguments or as --flag=value") {
  const auto check_all = [](const cli::Options& o) {
    CHECK(o.model_dir == fs::path("/m"));
    CHECK(o.prompt == "東京 = Tokyo");
    CHECK(o.max_tokens == 7);
    CHECK(o.sampling.temperature == 0.7f);
    CHECK(o.sampling.top_k == 40);
    CHECK(o.sampling.top_p == 0.9f);
    CHECK(o.sampling.repetition_penalty == 1.3f);
    CHECK(o.sampling.seed == 18446744073709551615ull);
    CHECK(o.threads == 4);
    CHECK(o.ignore_eos);
    CHECK(o.print_ids);
    CHECK_FALSE(o.help);
  };
  check_all(parse({"--model", "/m", "--prompt", "東京 = Tokyo", "--max-tokens", "7",
                   "--temperature", "0.7", "--top-k", "40", "--top-p", "0.9",
                   "--repetition-penalty", "1.3", "--seed", "18446744073709551615", "--threads",
                   "4", "--ignore-eos", "--print-ids"}));
  check_all(parse({"--print-ids", "--ignore-eos", "--threads=4", "--seed=18446744073709551615",
                   "--repetition-penalty=1.3", "--top-p=0.9", "--top-k=40", "--temperature=0.7",
                   "--max-tokens=7", "--prompt=東京 = Tokyo", "--model=/m"}));
}

TEST_CASE("parse_args: boundary values are accepted") {
  const cli::Options o = parse({"--model", "m", "--prompt", "p", "--max-tokens", "0",
                                "--temperature", "0", "--top-k", "0", "--top-p", "1",
                                "--repetition-penalty", "0.5", "--seed", "0", "--threads", "0"});
  CHECK(o.max_tokens == 0);
  CHECK(o.sampling.top_p == 1.0f);
  CHECK(o.sampling.repetition_penalty == 0.5f);
  CHECK(o.sampling.seed == 0u);
  CHECK(parse({"--model", "m", "--prompt", "p", "--top-p", "1e-9"}).sampling.top_p == 1e-9f);
}

TEST_CASE("parse_args: a repeated flag keeps its last value") {
  const cli::Options o =
      parse({"--model", "a", "--prompt", "p", "--max-tokens", "1", "--max-tokens", "2", "--model",
             "b"});
  CHECK(o.max_tokens == 2);
  CHECK(o.model_dir == fs::path("b"));
}

TEST_CASE("parse_args: an unknown flag is rejected by name") {
  CHECK(parse_error({"--model", "m", "--prompt", "p", "--bogus"}) == "unknown option '--bogus'");
  CHECK(parse_error({"--bogus=3", "--model", "m", "--prompt", "p"}) == "unknown option '--bogus'");
  CHECK(parse_error({"--model", "m", "--prompt", "p", "--max_tokens", "3"}) ==
        "unknown option '--max_tokens'");
  // Single-dash spellings are not aliases; only -h is.
  CHECK(parse_error({"-m", "x"}) == "unexpected argument '-m'");
  CHECK(parse_error({"--model", "m", "--prompt", "p", "stray"}) == "unexpected argument 'stray'");
}

TEST_CASE("parse_args: a flag missing its value is an error, not a crash") {
  for (const FlagSpec& f : kFlags) {
    if (!f.takes_value) continue;
    const std::string expected = std::string(f.name) + " needs a value";
    CAPTURE(expected);
    CHECK(parse_error({f.name}) == expected);                 // last on the line
    CHECK(parse_error({f.name, "--print-ids"}) == expected);  // followed by another flag
  }
  // The = form is how a value starting with "--" gets through.
  CHECK(parse({"--model", "m", "--prompt=--print-ids"}).prompt == "--print-ids");
  // A single dash is a value, so a negative number reaches the range check.
  CHECK(parse({"--model", "m", "--prompt", "-5"}).prompt == "-5");
  CHECK(parse_error({"--max-tokens", "-1"}) == "--max-tokens must be an integer >= 0, got '-1'");
}

TEST_CASE("parse_args: switches take no value") {
  CHECK(parse_error({"--ignore-eos=yes"}) == "--ignore-eos takes no value");
  CHECK(parse_error({"--print-ids=1"}) == "--print-ids takes no value");
  CHECK(parse_error({"--help=1"}) == "--help takes no value");
}

TEST_CASE("parse_args: malformed or out-of-range values name the flag and the value") {
  struct Bad {
    std::string_view flag;
    std::string_view value;
    std::string_view requirement;
  };
  const Bad cases[] = {
      {"--max-tokens", "abc", "an integer >= 0"},
      {"--max-tokens", "12abc", "an integer >= 0"},
      {"--max-tokens", "1.5", "an integer >= 0"},
      {"--max-tokens", "", "an integer >= 0"},
      {"--max-tokens", "-1", "an integer >= 0"},
      {"--max-tokens", "99999999999999999999", "an integer >= 0"},
      {"--temperature", "warm", "a number >= 0"},
      {"--temperature", "-0.5", "a number >= 0"},
      {"--temperature", "nan", "a number >= 0"},
      {"--temperature", "inf", "a number >= 0"},
      {"--temperature", "1e999", "a number >= 0"},
      {"--temperature", "0.7x", "a number >= 0"},
      {"--top-k", "-1", "an integer >= 0"},
      {"--top-k", "2.5", "an integer >= 0"},
      {"--top-p", "0", "a number in (0, 1]"},
      {"--top-p", "-0.1", "a number in (0, 1]"},
      {"--top-p", "1.5", "a number in (0, 1]"},
      {"--top-p", "nan", "a number in (0, 1]"},
      {"--repetition-penalty", "0", "a number > 0"},
      {"--repetition-penalty", "-1", "a number > 0"},
      {"--seed", "-1", "an integer >= 0"},
      {"--seed", "18446744073709551616", "an integer >= 0"},
      {"--seed", "0x10", "an integer >= 0"},
      {"--threads", "-1", "an integer >= 0"},
      {"--threads", "4294967296", "an integer >= 0"},
      {"--threads", "many", "an integer >= 0"},
  };
  for (const Bad& c : cases) {
    const std::string arg = std::string(c.flag) + "=" + std::string(c.value);
    const std::string expected = std::string(c.flag) + " must be " + std::string(c.requirement) +
                                 ", got '" + std::string(c.value) + "'";
    CAPTURE(arg);
    CHECK(parse_error({"--model", "m", "--prompt", "p", arg}) == expected);
  }
}

TEST_CASE("parse_args: --model and --prompt are required and non-empty") {
  CHECK(parse_error({}) == "missing required --model <dir>");
  CHECK(parse_error({"--prompt", "p"}) == "missing required --model <dir>");
  CHECK(parse_error({"--model", "m"}) == "missing required --prompt <text>");
  CHECK(parse_error({"--model", "m", "--prompt", ""}) == "--prompt must not be empty");
  CHECK(parse_error({"--model=", "--prompt", "p"}) == "--model must not be empty");
}

TEST_CASE("parse_args: --help stops parsing, so nothing else is required") {
  CHECK(parse({"--help"}).help);
  CHECK(parse({"-h"}).help);
  CHECK(parse({"--model", "m", "--help"}).help);
  CHECK(parse({"--help", "--bogus", "--max-tokens", "nope"}).help);
  // Anything malformed before it is still reported.
  CHECK(parse_error({"--bogus", "--help"}) == "unknown option '--bogus'");
}

TEST_CASE("the usage text lists exactly the flags the parser accepts, with their defaults") {
  std::set<std::string> documented;
  std::istringstream lines{std::string(cli::kUsage)};
  for (std::string line; std::getline(lines, line);) {
    if (line.starts_with("  --")) documented.insert(line.substr(2, line.find(' ', 2) - 2));
  }

  std::set<std::string> known;
  for (const FlagSpec& f : kFlags) {
    const std::string name(f.name);
    known.insert(name);
    CAPTURE(name);
    CHECK(documented.count(name) == 1u);
    // Whatever else the parser says about it, it must not call it unknown.
    CHECK_FALSE(contains(parse_error({f.name}), "unknown option"));
  }
  for (const std::string& name : documented) {
    CAPTURE(name);
    CHECK(known.count(name) == 1u);
  }

  // The defaults the usage advertises are the ones "defaults are the
  // documented ones" checks the parser against.
  const std::pair<std::string_view, std::string_view> defaults[] = {
      {"--max-tokens", "64"}, {"--temperature", "0"}, {"--top-k", "0"},  {"--top-p", "1"},
      {"--repetition-penalty", "1"}, {"--seed", "42"}, {"--threads", "0"},
  };
  for (const auto& [flag, value] : defaults) {
    const std::string line = line_containing(std::string(cli::kUsage), std::string(flag) + " ");
    CAPTURE(line);
    CHECK(contains(line, "(default " + std::string(value) + ")"));
  }
}

// ===========================================================================
// 1b. The model-directory check
// ===========================================================================

TEST_CASE("check_model_dir: a missing or incomplete directory is a readable error") {
  const fs::path root = make_temp_dir("model_dir_check");
  const auto where = [](const fs::path& p) { return "--model '" + p.string() + "'"; };

  const fs::path nowhere = root / "does-not-exist";
  CHECK(model_dir_error(nowhere) == where(nowhere) + ": no such directory");

  const fs::path file = root / "a-file";
  write_file(file);
  CHECK(model_dir_error(file) == where(file) + ": not a directory");

  const fs::path dir = make_temp_dir("model_dir_check/model");
  CHECK(model_dir_error(dir) == where(dir) +
                                    " is not a model directory: missing config.json, "
                                    "tokenizer.json, model.safetensors");
  write_file(dir / "config.json");
  CHECK(model_dir_error(dir) ==
        where(dir) + " is not a model directory: missing tokenizer.json, model.safetensors");
  write_file(dir / "tokenizer.json");
  CHECK(model_dir_error(dir) ==
        where(dir) + " is not a model directory: missing model.safetensors");

  // Either a single file or a sharded checkpoint's index will do.
  write_file(dir / "model.safetensors.index.json");
  CHECK(model_dir_error(dir).empty());
  fs::remove(dir / "model.safetensors.index.json");
  write_file(dir / "model.safetensors");
  CHECK(model_dir_error(dir).empty());
}

// ===========================================================================
// 1c. Utf8Stream
// ===========================================================================

TEST_CASE("Utf8Stream: ASCII passes straight through") {
  cli::Utf8Stream s;
  CHECK(s.push("Hello") == "Hello");
  CHECK(s.push(", world") == ", world");
  CHECK(s.push("") == "");
  CHECK(s.pending() == 0u);
  CHECK(s.finish().empty());
}

TEST_CASE("Utf8Stream: holds a split character back until its last byte arrives") {
  // é = C3 A9, € = E2 82 AC, 😀 = F0 9F 98 80
  cli::Utf8Stream s;
  CHECK(s.push("caf\xC3") == "caf");
  CHECK(s.pending() == 1u);
  CHECK(s.push("\xA9!") == "\xC3\xA9!");

  CHECK(s.push("\xE2") == "");
  CHECK(s.push("\x82") == "");
  CHECK(s.push("\xAC") == "\xE2\x82\xAC");

  CHECK(s.push("\xF0\x9F") == "");
  CHECK(s.push("\x98") == "");
  CHECK(s.pending() == 3u);
  CHECK(s.push("\x80 ok") == "\xF0\x9F\x98\x80 ok");
  CHECK(s.finish().empty());
}

TEST_CASE("Utf8Stream: every way of cutting mixed text reassembles exactly, never mid-character") {
  const std::string text = "aé€😀 東京 naïve 👍🏽!";
  REQUIRE(is_valid_utf8(text));

  std::string first_failure;
  for (std::size_t i = 0; i <= text.size(); ++i) {
    for (std::size_t j = i; j <= text.size(); ++j) {
      cli::Utf8Stream s;
      const std::string chunks[] = {s.push(text.substr(0, i)), s.push(text.substr(i, j - i)),
                                    s.push(text.substr(j)), s.finish()};
      std::string joined;
      bool valid = true;
      for (const std::string& c : chunks) {
        joined += c;
        valid = valid && is_valid_utf8(c);
      }
      if ((joined != text || !valid) && first_failure.empty()) {
        first_failure = "cut at bytes " + std::to_string(i) + " and " + std::to_string(j);
      }
    }
  }
  CHECK_MESSAGE(first_failure.empty(), first_failure);

  // And one byte at a time.
  cli::Utf8Stream s;
  std::string joined;
  for (const char c : text) {
    const std::string chunk = s.push(std::string_view(&c, 1));
    CHECK(is_valid_utf8(chunk));
    joined += chunk;
  }
  joined += s.finish();
  CHECK(joined == text);
}

TEST_CASE("Utf8Stream: bytes that can never complete pass through instead of being held") {
  cli::Utf8Stream s;
  CHECK(s.push("\x80") == "\x80");                  // stray continuation byte
  CHECK(s.push("\xFF") == "\xFF");                  // never valid in UTF-8
  CHECK(s.push("\xC3" "a") == "\xC3" "a");          // lead byte cut short by ASCII
  CHECK(s.push("\xC3\xA9\xA9") == "\xC3\xA9\xA9");  // one continuation byte too many
  CHECK(s.pending() == 0u);

  // A character still incomplete when the stream ends comes out of finish(),
  // unchanged, so not a byte is lost.
  CHECK(s.push("ok\xE6\x9D") == "ok");
  CHECK(s.finish() == "\xE6\x9D");
  CHECK(s.pending() == 0u);
}

// ===========================================================================
// 1d. The generation loop, with a fake model
// ===========================================================================

TEST_CASE("generate: prefill once at position 0, then one token per call at the cache length") {
  const LoopRun run = run_loop({11, 12, 13, 14}, 4, -1);
  assert_equal(run.gen.ids, std::vector<std::int32_t>{11, 12, 13, 14}, "generated ids");
  CHECK(run.gen.stop == cli::StopReason::kMaxTokens);
  // Each token is handed out before the next forward pass (that is the
  // streaming), and the last one is never fed back.
  CHECK(run.log ==
        "forward [5 6 7] at 0; token 11; forward [11] at 3; token 12; forward [12] at 4; "
        "token 13; forward [13] at 5; token 14");
}

TEST_CASE("generate: stops at eos without reporting it") {
  const LoopRun run = run_loop({11, 9, 12}, 10, 9);
  assert_equal(run.gen.ids, std::vector<std::int32_t>{11}, "generated ids");
  CHECK(run.gen.stop == cli::StopReason::kEos);
  CHECK(run.log == "forward [5 6 7] at 0; token 11; forward [11] at 3");

  const LoopRun immediate = run_loop({9}, 10, 9);
  CHECK(immediate.gen.ids.empty());
  CHECK(immediate.gen.stop == cli::StopReason::kEos);
  CHECK(immediate.log == "forward [5 6 7] at 0");
}

TEST_CASE("generate: eos_id -1 never stops early, so eos is an ordinary token") {
  const LoopRun run = run_loop({9, 9, 9}, 3, -1);
  assert_equal(run.gen.ids, std::vector<std::int32_t>{9, 9, 9}, "generated ids");
  CHECK(run.gen.stop == cli::StopReason::kMaxTokens);
}

TEST_CASE("generate: max_tokens 0 runs nothing, max_tokens 1 only the prefill") {
  const LoopRun none = run_loop({}, 0, -1);
  CHECK(none.gen.ids.empty());
  CHECK(none.gen.stop == cli::StopReason::kMaxTokens);
  CHECK(none.log.empty());

  const LoopRun one = run_loop({11}, 1, -1);
  assert_equal(one.gen.ids, std::vector<std::int32_t>{11}, "generated ids");
  CHECK(one.log == "forward [5 6 7] at 0; token 11");
}

TEST_CASE("generate: the sampler sees prompt + generated ids only when penalizing") {
  const LoopRun with = run_loop({11, 12, 13}, 3, -1, true);
  REQUIRE(with.histories.size() == 3u);
  CHECK(with.histories[0] == "5 6 7");
  CHECK(with.histories[1] == "5 6 7 11");
  CHECK(with.histories[2] == "5 6 7 11 12");

  const LoopRun without = run_loop({11, 12, 13}, 3, -1, false);
  REQUIRE(without.histories.size() == 3u);
  for (const std::string& h : without.histories) CHECK(h.empty());
}

// ===========================================================================
// 1e. Streaming with the real tokenizer (no model needed)
// ===========================================================================

TEST_CASE("streaming decode_token through Utf8Stream reproduces a one-call decode") {
  const std::optional<Tokenizer> tok = load_tokenizer();
  if (!tok) return;

  const std::string_view texts[] = {
      "Tokyo (東京) is located in",
      "日本の首都は東京です。",
      "naïve café, résumé — ünïcödé",
      "emoji: 😀🍕👍🏽",
      "mixed 中文 and ASCII, Ελληνικά, русский",
  };
  std::int64_t partial_tokens = 0;
  for (const std::string_view text : texts) {
    const std::string label(text);
    CAPTURE(label);
    const std::vector<std::int32_t> ids = tok->encode(text);
    cli::Utf8Stream stream;
    std::string streamed;
    bool chunks_valid = true;
    for (const std::int32_t id : ids) {
      const std::string piece = tok->decode_token(id);
      if (!is_valid_utf8(piece)) ++partial_tokens;
      const std::string chunk = stream.push(piece);
      chunks_valid = chunks_valid && is_valid_utf8(chunk);
      streamed += chunk;
    }
    streamed += stream.finish();
    CHECK(chunks_valid);
    CHECK(streamed == tok->decode(ids));
    CHECK(streamed == label);
  }
  // Otherwise no token split a character and the buffering went untested.
  CHECK(partial_tokens > 0);
}

// ===========================================================================
// 2. The binary: exit statuses and messages
// ===========================================================================

TEST_CASE("minillm --help prints the usage to stdout and exits 0") {
  if (!have_cli()) return;
  for (const char* flag : {"--help", "-h"}) {
    CAPTURE(flag);
    const RunResult r = run_cli({flag});
    CHECK(r.exit_code == 0);
    CHECK(r.out == std::string(cli::kUsage));
    CHECK(r.err.empty());
  }
}

TEST_CASE("minillm rejects a bad command line with a message and exit status 2") {
  if (!have_cli()) return;
  struct Case {
    std::vector<std::string> args;
    std::string message;
  };
  const Case cases[] = {
      {{}, "missing required --model <dir>"},
      {{"--model", "m", "--prompt", "p", "--bogus"}, "unknown option '--bogus'"},
      {{"--prompt", "p", "--model"}, "--model needs a value"},
      {{"--model", "m", "--prompt", "p", "--max-tokens", "lots"},
       "--max-tokens must be an integer >= 0, got 'lots'"},
  };
  for (const Case& c : cases) {
    CAPTURE(c.message);
    const RunResult r = run_cli(c.args);
    CHECK_FALSE(r.signaled);
    CHECK(r.exit_code == 2);
    CHECK(r.out.empty());
    CHECK(r.err.starts_with("minillm: " + c.message + "\n"));
  }
}

TEST_CASE("minillm reports a missing or incomplete --model directory and exits 1") {
  if (!have_cli()) return;
  const fs::path nowhere = make_temp_dir("cli_model_dir") / "nope";
  const fs::path empty = make_temp_dir("cli_model_dir/empty");
  struct Case {
    fs::path dir;
    std::string message;
  };
  for (const Case& c : {Case{nowhere, "no such directory"},
                        Case{empty, "missing config.json, tokenizer.json, model.safetensors"}}) {
    CAPTURE(c.message);
    const RunResult r = run_cli({"--model", c.dir.string(), "--prompt", "hi"});
    CHECK_FALSE(r.signaled);
    CHECK(r.exit_code == 1);
    CHECK(r.out.empty());
    CHECK(r.err.starts_with("minillm: error: --model '" + c.dir.string() + "'"));
    CHECK(contains(r.err, c.message));
  }
}

TEST_CASE("minillm turns a failure inside the engine into a message, not an escaped exception") {
  if (!have_cli()) return;
  // Every file is present, so the directory check passes, but none parses.
  const fs::path dir = make_temp_dir("cli_garbage_model");
  write_file(dir / "config.json", "{ this is not json");
  write_file(dir / "tokenizer.json");
  write_file(dir / "model.safetensors");
  const RunResult r = run_cli({"--model", dir.string(), "--prompt", "hi"});
  // An exception escaping main() would be std::terminate, i.e. SIGABRT.
  CHECK_FALSE(r.signaled);
  CHECK(r.exit_code == 1);
  CHECK(r.out.empty());
  CHECK(r.err.starts_with("minillm: error: "));
}

// ===========================================================================
// 3. End to end
// ===========================================================================

TEST_CASE("end to end: greedy decoding reproduces the oracle's tokens for every prompt") {
  const EndToEnd* e2e = e2e_or_skip();
  if (!e2e) return;
  for (std::size_t i = 0; i < kOraclePrompts.size(); ++i) {
    const std::string base = "model/p" + std::to_string(i) + "_";
    const std::string label = "prompt " + std::to_string(i);
    const RunResult& run = e2e->oracle[i];

    // The prompt first: if this differs, the bug is in the tokenizer.
    const std::optional<std::vector<std::int32_t>> prompt_ids = printed_ids(run.err, "prompt ids");
    CHECK_MESSAGE(prompt_ids.has_value(), "no 'prompt ids:' line on stderr:\n" << run.err);
    if (prompt_ids) {
      assert_equal(*prompt_ids, oracle_npy(base + "input_ids.npy").as_i32_narrowing(),
                   label + " input ids");
    }
    assert_equal(generated_ids(run), oracle_npy(base + "greedy_ids.npy").as_i32_narrowing(),
                 label + " greedy ids");
  }
}

TEST_CASE("end to end: generation stops at eos, which is neither printed nor counted") {
  const EndToEnd* e2e = e2e_or_skip();
  if (!e2e) return;
  const std::optional<Tokenizer> tok = load_tokenizer();
  if (!tok) return;

  // Prompt 1's oracle continuation runs straight through <|endoftext|>:
  // generate.py's loop never stops early.
  const std::int32_t eos = config_eos_id();
  const std::vector<std::int32_t> oracle = oracle_npy("model/p1_greedy_ids.npy").as_i32_narrowing();
  const auto at = std::find(oracle.begin(), oracle.end(), eos);
  REQUIRE_MESSAGE(at != oracle.end(), "oracle prompt 1 no longer generates eos (id "
                                          << eos << "); this test needs a prompt that does");
  const std::vector<std::int32_t> expected(oracle.begin(), at);

  const RunResult& run = e2e->stop_at_eos;
  assert_equal(generated_ids(run), expected, "prompt 1 up to eos");
  CHECK(contains(run.err, "stopped at eos after " + std::to_string(expected.size()) + " tokens"));
  CHECK(run.out == tok->decode(expected) + "\n");
}

TEST_CASE("end to end: streamed stdout is byte for byte a one-call decode of the ids") {
  const EndToEnd* e2e = e2e_or_skip();
  if (!e2e) return;
  const std::optional<Tokenizer> tok = load_tokenizer();
  if (!tok) return;

  struct Named {
    std::string name;
    const RunResult* run;
  };
  for (const Named& n : {Named{"prompt 0", &e2e->oracle[0]}, Named{"prompt 1", &e2e->oracle[1]},
                         Named{"prompt 2", &e2e->oracle[2]},
                         Named{"prompt 1 to eos", &e2e->stop_at_eos},
                         Named{"multi-byte prompt", &e2e->multibyte}}) {
    CAPTURE(n.name);
    const std::vector<std::int32_t> ids = generated_ids(*n.run);
    CHECK(n.run->out == tok->decode(ids) + "\n");
  }

  // The multi-byte run must actually have streamed split characters, or it
  // proves nothing beyond the ASCII runs.
  const std::vector<std::int32_t> ids = generated_ids(e2e->multibyte);
  const std::string text = tok->decode(ids);
  CHECK(std::any_of(text.begin(), text.end(),
                    [](char c) { return static_cast<unsigned char>(c) >= 0x80; }));
  CHECK(std::any_of(ids.begin(), ids.end(),
                    [&](std::int32_t id) { return !is_valid_utf8(tok->decode_token(id)); }));
  CHECK(is_valid_utf8(e2e->multibyte.out));
}
