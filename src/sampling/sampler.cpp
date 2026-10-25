// Sampler — owned by workflow 07-sampling.
//
// The stages run in the order sampler.h documents. Two things keep every draw
// reproducible across platforms and standard libraries:
//
//   * Candidates are ordered by (logit descending, id ascending), a strict
//     total order, and always walked in a canonical order — never in whatever
//     arrangement std::nth_element happened to leave behind, which differs
//     between libc++ and libstdc++.
//   * The draw maps raw std::mt19937_64 output (fully specified by the
//     standard) to [0, 1) by hand and walks the cumulative sum itself.
//     std::uniform_real_distribution and std::discrete_distribution are not
//     specified tightly enough to give the same answer everywhere.
//
// The only platform dependence left is std::exp's last ulp, which can move a
// draw only when it lands exactly on a probability boundary.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <locale>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "minillm/sampler.h"

namespace minillm {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument("SamplingParams: " + message);
}

/// Formats a float the way a person would write it (0.7, not 0.700000),
/// independent of the global locale.
std::string format_float(float v) {
  std::ostringstream os;
  os.imbue(std::locale::classic());
  os << v;
  return os.str();
}

/// (logit descending, id ascending). A strict total order as long as no logit
/// is NaN, so "the k largest" is a well-defined set even with ties.
struct ByLogitDesc {
  const float* logits;
  bool operator()(std::int32_t a, std::int32_t b) const {
    return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
  }
};

/// Same rule as ops::argmax: strictly greater wins, so ties go to the lowest id.
std::int32_t argmax(std::span<const float> logits) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) best = i;
  }
  return static_cast<std::int32_t>(best);
}

/// Stage 1. Each distinct id in `history` is penalized exactly once.
void apply_repetition_penalty(std::span<float> logits, std::span<const std::int32_t> history,
                              float penalty) {
  std::vector<std::int32_t> ids(history.begin(), history.end());
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  if (ids.empty()) return;

  const auto vocab_size = static_cast<std::int64_t>(logits.size());
  for (const std::int32_t id : {ids.front(), ids.back()}) {
    if (id < 0 || id >= vocab_size) {
      throw std::out_of_range("Sampler::sample: history id " + std::to_string(id) +
                              " out of range for vocab size " + std::to_string(vocab_size));
    }
  }

  float* const x = logits.data();
  for (const std::int32_t id : ids) {
    // Dividing a negative logit would make the token *more* likely.
    x[id] = x[id] > 0.0f ? x[id] / penalty : x[id] * penalty;
  }
}

/// 53 random bits from one engine output, as a double in [0, 1).
double uniform01(std::mt19937_64& rng) {
  return static_cast<double>(rng() >> 11) * 0x1.0p-53;
}

}  // namespace

void SamplingParams::validate() const {
  if (top_k < 0) fail("top_k must be >= 0, got " + std::to_string(top_k));
  // Negated comparisons so that NaN is rejected too.
  if (!(top_p > 0.0f)) fail("top_p must be > 0, got " + format_float(top_p));
  if (!(repetition_penalty > 0.0f)) {
    fail("repetition_penalty must be > 0, got " + format_float(repetition_penalty));
  }
}

std::string SamplingParams::to_string() const {
  return "temperature=" + format_float(temperature) + " top_k=" + std::to_string(top_k) +
         " top_p=" + format_float(top_p) +
         " repetition_penalty=" + format_float(repetition_penalty) +
         " seed=" + std::to_string(seed);
}

Sampler::Sampler(const SamplingParams& params) : params_(params), rng_(params.seed) {
  params_.validate();
}

std::int32_t Sampler::sample(std::span<float> logits, std::span<const std::int32_t> history) {
  if (logits.empty()) throw std::invalid_argument("Sampler::sample: logits is empty");
  if (logits.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::invalid_argument("Sampler::sample: " + std::to_string(logits.size()) +
                                " logits do not fit int32 token ids");
  }

  // 1. Repetition penalty — before the greedy short-circuit, so it applies there too.
  if (params_.repetition_penalty != 1.0f) {
    apply_repetition_penalty(logits, history, params_.repetition_penalty);
  }
  if (params_.temperature <= 0.0f) return argmax(logits);

  // 2. Temperature. A NaN logit becomes -inf, an impossible token, so that
  // every ordering below is a strict total order.
  for (float& v : logits) {
    v /= params_.temperature;
    if (std::isnan(v)) v = kNegInf;
  }

  float* const x = logits.data();
  const std::size_t n = logits.size();
  const std::size_t k = params_.top_k > 0 && static_cast<std::uint64_t>(params_.top_k) < n
                            ? static_cast<std::size_t>(params_.top_k)
                            : n;
  const bool top_p = params_.top_p < 1.0f;

  // Every token still in play, in index order until a filter needs them sorted.
  std::vector<std::int32_t> ids(n);
  std::iota(ids.begin(), ids.end(), 0);
  const ByLogitDesc order{x};

  // 3. Top-k: a selection, not a sort of the whole vocabulary.
  if (k < n) {
    std::nth_element(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(k), ids.end(), order);
    ids.resize(k);
  }
  // Top-p needs the candidates in descending order. Top-k alone only needs an
  // order that does not depend on how nth_element arranged them.
  if (k < n || top_p) std::sort(ids.begin(), ids.end(), order);

  float max_logit = kNegInf;
  for (const std::int32_t id : ids) max_logit = std::max(max_logit, x[id]);
  // Every candidate is -inf, or some are +inf: there is no proper
  // distribution, and the most likely token is the only sensible answer.
  if (!std::isfinite(max_logit)) {
    for (const std::int32_t id : ids) {
      if (x[id] == max_logit) return id;
    }
  }

  // Unnormalized softmax weights, written over the logits (they are scratch).
  double total = 0.0;
  for (const std::int32_t id : ids) {
    x[id] = std::exp(x[id] - max_logit);
    total += x[id];
  }

  // 4. Top-p: the shortest descending prefix holding top_p of the mass. The
  // loop body runs at least once, so at least one token is always kept.
  if (top_p) {
    const double threshold = static_cast<double>(params_.top_p) * total;
    double kept = 0.0;
    std::size_t n_kept = 0;
    do {
      kept += x[ids[n_kept++]];
    } while (n_kept < ids.size() && kept < threshold);
    ids.resize(n_kept);
    total = kept;
  }

  // 5. One multinomial draw: the first token whose cumulative weight exceeds
  // u * total, so a zero-weight token can never be picked.
  const double target = uniform01(rng_) * total;
  double cumulative = 0.0;
  for (const std::int32_t id : ids) {
    cumulative += x[id];
    if (target < cumulative) return id;
  }
  return ids.back();  // unreachable: target < total, and this walk sums to exactly total
}

void Sampler::reset() { rng_.seed(params_.seed); }

}  // namespace minillm
