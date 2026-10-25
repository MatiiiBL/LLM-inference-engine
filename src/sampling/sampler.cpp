// Sampler — owned by workflow 07-sampling.
//
// STUB. The constructor works (so a Sampler can be held by the CLI before
// sampling exists) but sample() throws.

#include <stdexcept>

#include "minillm/sampler.h"

namespace minillm {

void SamplingParams::validate() const {
  throw std::logic_error("not implemented: SamplingParams::validate (workflow 07-sampling)");
}

std::string SamplingParams::to_string() const {
  throw std::logic_error("not implemented: SamplingParams::to_string (workflow 07-sampling)");
}

Sampler::Sampler(const SamplingParams& params) : params_(params), rng_(params.seed) {}

std::int32_t Sampler::sample(std::span<float>, std::span<const std::int32_t>) {
  throw std::logic_error("not implemented: Sampler::sample (workflow 07-sampling)");
}

void Sampler::reset() { rng_.seed(params_.seed); }

}  // namespace minillm
