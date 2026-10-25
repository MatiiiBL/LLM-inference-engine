// The `minillm` executable — owned by workflow 09-cli.
//
// Load a model, tokenize a prompt, generate, print. The continuation is
// streamed to stdout as it is generated, followed by one newline, and nothing
// else is ever written there: the banner, timings and --print-ids all go to
// stderr, so `minillm ... > out.txt` captures exactly the generated text.
//
// Exit status: 0 on success (and for --help), 2 for a malformed command line,
// 1 for anything that goes wrong after parsing. No exception escapes main().
//
// The pieces live in private headers next to this file so the test can reach
// them: args.h (flags, usage, the model-directory check), utf8_stream.h and
// generate.h (the prefill/decode loop).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "args.h"
#include "generate.h"
#include "minillm/config.h"
#include "minillm/loader.h"
#include "minillm/model.h"
#include "minillm/sampler.h"
#include "minillm/tokenizer.h"
#include "utf8_stream.h"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

void write_stdout(const std::string& bytes) {
  if (bytes.empty()) return;
  std::fwrite(bytes.data(), 1, bytes.size(), stdout);
  std::fflush(stdout);
}

void print_ids(const char* label, std::span<const std::int32_t> ids) {
  std::fprintf(stderr, "%s:", label);
  for (const std::int32_t id : ids) std::fprintf(stderr, " %d", id);
  std::fprintf(stderr, "\n");
}

int run(const minillm::cli::Options& opts) {
  using namespace minillm;

  // Cheap checks first, so a typo never waits on a 500 MB weight load.
  cli::check_model_dir(opts.model_dir);
  opts.sampling.validate();
  // opts.threads is parsed and range-checked but goes nowhere yet, as --help
  // says: nothing in include/ lets the CLI hand a thread count to the kernels
  // (Model takes none, the ops take no pool, and runtime.h has no
  // process-wide pool).

  const Clock::time_point t_load = Clock::now();
  ModelConfig cfg = ModelConfig::from_json_file(opts.model_dir / "config.json");
  const Tokenizer tokenizer = Tokenizer::from_dir(opts.model_dir);
  const std::vector<std::int32_t> prompt_ids = tokenizer.encode(opts.prompt);

  const auto n_prompt = static_cast<std::int64_t>(prompt_ids.size());
  if (opts.max_tokens > cfg.max_position_embeddings - n_prompt) {
    throw std::runtime_error("the prompt (" + std::to_string(n_prompt) + " tokens) plus " +
                             "--max-tokens " + std::to_string(opts.max_tokens) +
                             " exceeds the model's context of " +
                             std::to_string(cfg.max_position_embeddings) + " tokens");
  }
  // config.json is what transformers' generate() stops on; tokenizer_config.json
  // is only a fallback for checkpoints whose config does not name one.
  const std::int32_t eos_id =
      opts.ignore_eos ? -1 : (cfg.eos_token_id >= 0 ? cfg.eos_token_id : tokenizer.eos_id());

  Model model(std::move(cfg), load_checkpoint(opts.model_dir), n_prompt + opts.max_tokens);
  const ModelConfig& mc = model.config();
  std::fprintf(stderr,
               "minillm: loaded %s (%s, %lld layers, hidden %lld, vocab %lld) in %.2f s\n",
               opts.model_dir.string().c_str(), mc.model_type.c_str(),
               static_cast<long long>(mc.num_hidden_layers),
               static_cast<long long>(mc.hidden_size), static_cast<long long>(mc.vocab_size),
               seconds_between(t_load, Clock::now()));
  if (opts.print_ids) print_ids("prompt ids", prompt_ids);

  Sampler sampler(opts.sampling);
  cli::Utf8Stream stream;
  const Clock::time_point t_start = Clock::now();
  Clock::time_point t_first = t_start;
  bool first = true;

  const cli::Generation gen = cli::generate(
      model, sampler, prompt_ids, opts.max_tokens, eos_id,
      opts.sampling.repetition_penalty != 1.0f, [&](std::int32_t id) {
        if (first) {
          t_first = Clock::now();
          first = false;
        }
        write_stdout(stream.push(tokenizer.decode_token(id)));
      });
  write_stdout(stream.finish() + "\n");
  const Clock::time_point t_end = Clock::now();

  const auto n_gen = static_cast<long long>(gen.ids.size());
  std::fprintf(stderr, "minillm: stopped at %s after %lld tokens; prompt %lld tokens",
               gen.stop == cli::StopReason::kEos ? "eos" : "max-tokens", n_gen,
               static_cast<long long>(n_prompt));
  if (n_gen > 0) {
    std::fprintf(stderr, ", first token after %.2f s", seconds_between(t_start, t_first));
  }
  if (n_gen > 1) {
    std::fprintf(stderr, ", %.1f tok/s after that",
                 static_cast<double>(n_gen - 1) / seconds_between(t_first, t_end));
  }
  std::fprintf(stderr, "\n");
  if (opts.print_ids) print_ids("generated ids", gen.ids);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::vector<std::string_view> args(argv + 1, argv + argc);
    minillm::cli::Options opts;
    try {
      opts = minillm::cli::parse_args(args);
    } catch (const minillm::cli::UsageError& e) {
      std::fprintf(stderr, "minillm: %s\nRun 'minillm --help' for usage.\n", e.what());
      return 2;
    }
    if (opts.help) {
      std::fwrite(minillm::cli::kUsage.data(), 1, minillm::cli::kUsage.size(), stdout);
      return 0;
    }
    return run(opts);
  } catch (const std::exception& e) {
    std::fflush(stdout);
    std::fprintf(stderr, "minillm: error: %s\n", e.what());
    return 1;
  } catch (...) {
    std::fflush(stdout);
    std::fprintf(stderr, "minillm: error: unknown exception\n");
    return 1;
  }
}
