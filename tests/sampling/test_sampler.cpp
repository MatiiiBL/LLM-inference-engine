// Workflow 07-sampling owns this file.
//
// PLACEHOLDER. Sampler::sample is a stub.
//
// What this must cover when it is done:
//   * temperature <= 0 returns argmax (compare against ops::argmax)
//   * the same seed and logits give the same id, every time; a different seed
//     generally gives a different one
//   * top_k = 1 is greedy regardless of temperature
//   * top_p = 1 with a huge temperature approaches a uniform draw; measure the
//     distribution over many draws, with a fixed seed so the test is not flaky
//   * top_k and top_p never select a token outside the kept set — build logits
//     where the answer is known and check the id is in it
//   * the repetition penalty divides positive logits and multiplies negative
//     ones, applied to distinct ids in the history only
//   * the stage order matches the header: penalty, temperature, top-k, top-p
//   * reset() restores the sequence

#include <doctest/doctest.h>

#include <stdexcept>
#include <vector>

#include "common/oracle.h"
#include "minillm/sampler.h"

using namespace minillm;
using namespace minillm::test;

TEST_CASE("SamplingParams defaults are greedy") {
  const SamplingParams params;
  CHECK(params.temperature == 0.0f);
  CHECK(params.top_k == 0);
  CHECK(params.top_p == 1.0f);
  CHECK(params.repetition_penalty == 1.0f);
  CHECK(params.seed == 42);
}

TEST_CASE("a Sampler can be constructed and re-seeded before sampling exists") {
  SamplingParams params;
  params.seed = 7;
  Sampler sampler(params);
  CHECK(sampler.params().seed == 7);
  sampler.reset();  // must not throw: the RNG is real even though sample() is not
}

TEST_CASE("sampling stubs are linked and announce themselves") {
  MESSAGE("SKIP: the sampler is a stub — workflow 07-sampling implements it");
  note_skip("sampling: distribution tests pending workflow 07-sampling");

  CHECK_THROWS_AS(SamplingParams().validate(), std::logic_error);
  CHECK_THROWS_AS(SamplingParams().to_string(), std::logic_error);

  Sampler sampler(SamplingParams{});
  std::vector<float> logits{0.1f, 0.2f, 0.3f};
  CHECK_THROWS_AS(sampler.sample(logits, std::vector<std::int32_t>{}), std::logic_error);
}
