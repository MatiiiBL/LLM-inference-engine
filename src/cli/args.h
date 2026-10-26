// Command-line parsing for the `minillm` executable.
//
// Private to src/cli/. Header-only because src/cli/main.cpp is the only source
// file the minillm target compiles, and tests/cli/test_cli.cpp includes this
// header directly so the parser can be tested without spawning a process.

#ifndef MINILLM_SRC_CLI_ARGS_H
#define MINILLM_SRC_CLI_ARGS_H

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

#include "minillm/sampler.h"
#include "minillm/tensor.h"

namespace minillm::cli {

/// Everything the command line controls. The defaults are the documented ones;
/// SamplingParams' own defaults (greedy, seed 42) already match them.
struct Options {
  std::filesystem::path model_dir;
  std::string prompt;
  std::int64_t max_tokens = 64;
  SamplingParams sampling;
  int threads = 0;
  DType weights = DType::F32;  ///< F32 or Q8_0 (quantize_weights_q8_0 at load)
  bool ignore_eos = false;
  bool print_ids = false;
  bool help = false;
};

/// A malformed command line. The message names the offending flag.
class UsageError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Keep in sync with parse_args() below; the test checks that every flag listed
// here is one the parser accepts, and the other way round.
inline constexpr std::string_view kUsage =
    "usage: minillm --model <dir> --prompt <text> [options]\n"
    "\n"
    "  --model <dir>              config.json, tokenizer.json, model.safetensors\n"
    "  --prompt <text>            prompt to complete\n"
    "  --max-tokens <n>           tokens to generate       (default 64)\n"
    "  --temperature <f>          0 means greedy           (default 0)\n"
    "  --top-k <n>                0 disables               (default 0)\n"
    "  --top-p <f>                1 disables               (default 1)\n"
    "  --repetition-penalty <f>                            (default 1)\n"
    "  --seed <n>                                          (default 42)\n"
    "  --threads <n>              threads to use, 0 = auto (default 0)\n"
    "  --weights <f32|q8_0>       weight format            (default f32)\n"
    "  --ignore-eos               keep generating past the end-of-text token\n"
    "  --print-ids                print prompt and generated token ids to stderr\n"
    "  --help\n"
    "\n"
    "The continuation is streamed to stdout; everything else goes to stderr.\n"
    "Any flag that takes a value also accepts --flag=value.\n"
    "--threads 0 uses one thread per hardware thread. Where cores differ (Apple\n"
    "silicon), only work big enough to gain from the efficiency cores, such as a\n"
    "long prompt, uses more threads than the performance cores have. The output\n"
    "is the same at any --threads; only the speed changes.\n"
    "--weights q8_0 rounds the projections and the embedding to 8 bits at load\n"
    "(143 MB instead of 538 MB for SmolLM2-135M); arithmetic stays f32.\n";

namespace detail {

/// Parses all of `text` as a float, accepting what std::from_chars does: an
/// optional '-', then decimal digits with an optional fraction and exponent.
/// std::from_chars itself has no float overload in the libc++ that Xcode 16
/// ships, so this goes through strtof, after rejecting the extra syntax strtof
/// would take: leading whitespace, a leading '+', hex floats, inf and nan.
inline bool parse_float(std::string_view text, float& value) {
  if (text.empty() || text.front() == '+') return false;
  if (text.find_first_not_of("0123456789.eE+-") != std::string_view::npos) return false;
  const std::string s(text);
  char* end = nullptr;
  errno = 0;
  value = std::strtof(s.c_str(), &end);
  return end == s.c_str() + s.size() && errno != ERANGE;
}

/// Parses all of `text` as a T and checks it with `ok`. Anything else — junk,
/// trailing characters, overflow, NaN or infinity, a value `ok` rejects — is
/// a UsageError of the form "--flag must be <requirement>, got '<text>'".
template <class T, class Pred>
T parse_number(std::string_view flag, std::string_view text, std::string_view requirement,
               Pred ok) {
  T value{};
  bool valid = false;
  if constexpr (std::is_floating_point_v<T>) {
    static_assert(std::is_same_v<T, float>, "parse_float only handles float");
    valid = parse_float(text, value) && std::isfinite(value);
  } else {
    const char* first = text.data();
    const char* last = first + text.size();
    const auto [ptr, ec] = std::from_chars(first, last, value);
    valid = ec == std::errc{} && ptr == last;
  }
  if (!valid || !ok(value)) {
    throw UsageError(std::string(flag) + " must be " + std::string(requirement) + ", got '" +
                     std::string(text) + "'");
  }
  return value;
}

}  // namespace detail

/// Parses the arguments after argv[0]. Flags take their value either as the
/// next argument or as --flag=value; the next argument counts as missing if it
/// starts with "--", so a prompt that does must use the = form. A repeated
/// flag keeps its last value. --help (or -h) stops parsing and skips the
/// required-flag checks. Throws UsageError on anything malformed.
inline Options parse_args(std::span<const std::string_view> args) {
  Options opts;
  bool have_model = false;
  bool have_prompt = false;

  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (arg == "-h") {
      opts.help = true;
      return opts;
    }
    if (!arg.starts_with("--")) {
      throw UsageError("unexpected argument '" + std::string(arg) + "'");
    }

    std::string_view flag = arg;
    std::optional<std::string_view> inline_value;
    if (const std::size_t eq = arg.find('='); eq != std::string_view::npos) {
      flag = arg.substr(0, eq);
      inline_value = arg.substr(eq + 1);
    }

    const auto no_value = [&] {
      if (inline_value) throw UsageError(std::string(flag) + " takes no value");
    };
    const auto value = [&]() -> std::string_view {
      if (inline_value) return *inline_value;
      if (i + 1 >= args.size() || args[i + 1].starts_with("--")) {
        throw UsageError(std::string(flag) + " needs a value");
      }
      return args[++i];
    };
    const auto non_negative = [](auto v) { return v >= 0; };
    const auto positive = [](auto v) { return v > 0; };

    if (flag == "--help") {
      no_value();
      opts.help = true;
      return opts;
    } else if (flag == "--model") {
      opts.model_dir = value();
      have_model = true;
    } else if (flag == "--prompt") {
      opts.prompt = value();
      have_prompt = true;
    } else if (flag == "--max-tokens") {
      opts.max_tokens =
          detail::parse_number<std::int64_t>(flag, value(), "an integer >= 0", non_negative);
    } else if (flag == "--temperature") {
      opts.sampling.temperature =
          detail::parse_number<float>(flag, value(), "a number >= 0", non_negative);
    } else if (flag == "--top-k") {
      opts.sampling.top_k =
          detail::parse_number<std::int64_t>(flag, value(), "an integer >= 0", non_negative);
    } else if (flag == "--top-p") {
      opts.sampling.top_p = detail::parse_number<float>(
          flag, value(), "a number in (0, 1]", [](float v) { return v > 0.0f && v <= 1.0f; });
    } else if (flag == "--repetition-penalty") {
      opts.sampling.repetition_penalty =
          detail::parse_number<float>(flag, value(), "a number > 0", positive);
    } else if (flag == "--seed") {
      opts.sampling.seed = detail::parse_number<std::uint64_t>(
          flag, value(), "an integer >= 0", [](std::uint64_t) { return true; });
    } else if (flag == "--threads") {
      opts.threads = detail::parse_number<int>(flag, value(), "an integer >= 0", non_negative);
    } else if (flag == "--weights") {
      const std::string_view v = value();
      if (v == "f32") {
        opts.weights = DType::F32;
      } else if (v == "q8_0") {
        opts.weights = DType::Q8_0;
      } else {
        throw UsageError("--weights must be f32 or q8_0, got '" + std::string(v) + "'");
      }
    } else if (flag == "--ignore-eos") {
      no_value();
      opts.ignore_eos = true;
    } else if (flag == "--print-ids") {
      no_value();
      opts.print_ids = true;
    } else {
      throw UsageError("unknown option '" + std::string(flag) + "'");
    }
  }

  if (!have_model) throw UsageError("missing required --model <dir>");
  if (opts.model_dir.empty()) throw UsageError("--model must not be empty");
  if (!have_prompt) throw UsageError("missing required --prompt <text>");
  if (opts.prompt.empty()) throw UsageError("--prompt must not be empty");
  return opts;
}

/// Checks that `dir` holds the files a model needs before anything tries to
/// load it, so a typo or a half-copied checkpoint produces one readable line
/// rather than whichever loader error happens to trip first. Throws
/// std::runtime_error listing every missing file at once.
inline void check_model_dir(const std::filesystem::path& dir) {
  namespace fs = std::filesystem;
  std::error_code ec;
  const std::string where = "--model '" + dir.string() + "'";
  if (!fs::exists(dir, ec)) throw std::runtime_error(where + ": no such directory");
  if (!fs::is_directory(dir, ec)) throw std::runtime_error(where + ": not a directory");

  std::vector<std::string> missing;
  for (const char* name : {"config.json", "tokenizer.json"}) {
    if (!fs::is_regular_file(dir / name, ec)) missing.emplace_back(name);
  }
  // A sharded checkpoint has an index instead of model.safetensors; the
  // loader accepts either.
  if (!fs::is_regular_file(dir / "model.safetensors", ec) &&
      !fs::is_regular_file(dir / "model.safetensors.index.json", ec)) {
    missing.emplace_back("model.safetensors");
  }
  if (missing.empty()) return;

  std::string list;
  for (const std::string& name : missing) {
    if (!list.empty()) list += ", ";
    list += name;
  }
  throw std::runtime_error(where + " is not a model directory: missing " + list);
}

}  // namespace minillm::cli

#endif  // MINILLM_SRC_CLI_ARGS_H
