// The generation loop behind `minillm`.
//
// Separate from main() so the test can drive it with a fake model: the test
// target does not link minillm_model. That is the only reason it is a
// template — the executable instantiates it with minillm::Model and
// minillm::Sampler, and the fakes only need the members used here:
//
//   model.forward(std::span<const std::int32_t>, std::int64_t start_pos)
//       -> std::span<const float>                     logits of the last token
//   model.kv_cache().seq_len() -> std::int64_t
//   sampler.sample(std::span<float>, std::span<const std::int32_t>) -> std::int32_t
//
// Private to src/cli/; header-only for the same reason as args.h.

#ifndef MINILLM_SRC_CLI_GENERATE_H
#define MINILLM_SRC_CLI_GENERATE_H

#include <cstdint>
#include <span>
#include <vector>

namespace minillm::cli {

enum class StopReason { kMaxTokens, kEos };

struct Generation {
  std::vector<std::int32_t> ids;  ///< every id sampled, minus a terminating eos
  StopReason stop = StopReason::kMaxTokens;
};

/// Prefills the whole prompt in one forward() call at start_pos 0, then
/// decodes one token per call at start_pos = model.kv_cache().seq_len().
/// on_token(id) runs as soon as each id is sampled, before the next forward
/// pass, which is what makes the output stream.
///
/// Stops after `max_tokens` ids, or as soon as `eos_id` is sampled; the eos
/// itself is neither passed to on_token nor returned. eos_id < 0 never stops
/// early. The last id is never fed back through the model, since nothing
/// would read its logits.
///
/// With `penalize_history` the sampler sees the prompt plus everything
/// generated so far, which is what the repetition penalty reads; without it,
/// an empty span, as Sampler::sample asks for when the penalty is 1.
template <class ModelT, class SamplerT, class OnToken>
Generation generate(ModelT& model, SamplerT& sampler, std::span<const std::int32_t> prompt,
                    std::int64_t max_tokens, std::int32_t eos_id, bool penalize_history,
                    OnToken&& on_token) {
  Generation gen;
  if (max_tokens <= 0) return gen;

  std::vector<std::int32_t> history(prompt.begin(), prompt.end());
  // forward() hands out read-only logits; sample() uses its input as scratch.
  std::vector<float> scratch;

  std::span<const float> logits = model.forward(prompt, 0);
  for (;;) {
    scratch.assign(logits.begin(), logits.end());
    const std::int32_t id = sampler.sample(
        scratch, penalize_history ? std::span<const std::int32_t>(history)
                                  : std::span<const std::int32_t>());
    if (id == eos_id) {
      gen.stop = StopReason::kEos;
      break;
    }
    gen.ids.push_back(id);
    history.push_back(id);
    on_token(id);
    if (static_cast<std::int64_t>(gen.ids.size()) == max_tokens) break;
    logits = model.forward(std::span<const std::int32_t>(&id, 1), model.kv_cache().seq_len());
  }
  return gen;
}

}  // namespace minillm::cli

#endif  // MINILLM_SRC_CLI_GENERATE_H
