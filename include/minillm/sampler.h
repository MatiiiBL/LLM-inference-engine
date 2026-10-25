// minillm — sampler.h
//
// Turns a logits row into a token id. Seeded and deterministic: the same
// params, logits and history always produce the same id on every platform.
//
// THIS HEADER IS FROZEN. See docs/ARCHITECTURE.md.

#ifndef MINILLM_SAMPLER_H
#define MINILLM_SAMPLER_H

#include <cstdint>
#include <random>
#include <span>
#include <string>

namespace minillm {

/// Decoding parameters. The defaults are greedy decoding.
struct SamplingParams {
  /// <= 0 means greedy (argmax); the other knobs are then ignored.
  float temperature = 0.0f;
  /// Keep only the top_k highest-probability tokens. 0 disables.
  std::int64_t top_k = 0;
  /// Keep the smallest prefix whose cumulative probability >= top_p.
  /// Values >= 1 disable it.
  float top_p = 1.0f;
  /// > 1 discourages tokens already present in the history, 1 disables.
  float repetition_penalty = 1.0f;
  /// RNG seed. Fixed by default so runs are reproducible.
  std::uint64_t seed = 42;

  /// Throws std::invalid_argument if top_k < 0, top_p <= 0, or
  /// repetition_penalty <= 0.
  void validate() const;
  std::string to_string() const;
};

/// Single-sequence sampler.
///
/// sample() applies the stages below in this exact order — the same order as
/// transformers' default logits processors, so results can be compared:
///
///   1. repetition penalty: for each distinct id in `history`,
///        logits[id] = logits[id] > 0 ? logits[id] / p : logits[id] * p
///   2. temperature:   logits[i] /= temperature
///   3. top-k:         all but the k largest logits become -inf
///   4. top-p:         sort descending, keep the shortest prefix whose
///                     cumulative softmax probability >= top_p (always at
///                     least one token), the rest become -inf
///   5. softmax, then one multinomial draw from std::mt19937_64 seeded with
///      params().seed.
///
/// Step 1 is applied even when temperature <= 0 (greedy), so a greedy run
/// with a repetition penalty still behaves as expected.
class Sampler {
 public:
  explicit Sampler(const SamplingParams& params);

  /// Picks the next token.
  ///
  ///   logits   [vocab_size] F32, MODIFIED IN PLACE (used as scratch)
  ///   history  previously generated/prompt ids, for the repetition penalty;
  ///            pass an empty span when the penalty is 1.
  ///
  /// Throws std::invalid_argument if logits is empty.
  std::int32_t sample(std::span<float> logits, std::span<const std::int32_t> history);

  /// Re-seeds the RNG from params().seed, making a fresh run reproducible.
  void reset();

  const SamplingParams& params() const { return params_; }

 private:
  SamplingParams params_;
  std::mt19937_64 rng_;
};

}  // namespace minillm

#endif  // MINILLM_SAMPLER_H
