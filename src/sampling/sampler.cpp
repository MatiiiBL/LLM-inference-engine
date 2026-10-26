// Sampler — owned by workflow 07-sampling.
//
// The stages run in the order sampler.h documents. Two things keep every draw
// reproducible across platforms and standard libraries:
//
//   * Candidates are ordered by (value descending, id ascending), a strict
//     total order, and always summed and walked in a canonical order — never
//     in whatever arrangement std::nth_element happened to leave behind,
//     which differs between libc++ and libstdc++.
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

/// (value descending, id ascending). A strict total order as long as no value
/// is NaN, so "the k largest" is a well-defined set even with ties.
struct Descending {
  const float* values;
  bool operator()(std::int32_t a, std::int32_t b) const {
    return values[a] > values[b] || (values[a] == values[b] && a < b);
  }
};

/// Same rule as ops::argmax, which is torch.argmax's: the first NaN if there
/// is one; otherwise strictly greater wins, so ties go to the lowest id.
std::int32_t argmax(std::span<const float> logits) {
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (std::isnan(logits[i])) return static_cast<std::int32_t>(i);
  }
  std::size_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) best = i;
  }
  return static_cast<std::int32_t>(best);
}

/// The largest logit, or -inf if there is none. NaNs are skipped: std::max
/// keeps its first argument against one. Four running maxima rather than one
/// make the loop bound by throughput, not by one long chain of comparisons.
float largest(std::span<const float> logits) {
  float lanes[4] = {kNegInf, kNegInf, kNegInf, kNegInf};
  std::size_t i = 0;
  for (; i + 4 <= logits.size(); i += 4) {
    for (std::size_t j = 0; j < 4; ++j) lanes[j] = std::max(lanes[j], logits[i + j]);
  }
  for (; i < logits.size(); ++i) lanes[0] = std::max(lanes[0], logits[i]);
  return std::max(std::max(lanes[0], lanes[1]), std::max(lanes[2], lanes[3]));
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

/// Stage 4. Shrinks `ids` to the shortest prefix, in descending weight order,
/// whose weight reaches `threshold`, and returns that prefix's weight. The
/// first id is always kept.
///
/// Unless `ids` is already sorted, the order is established lazily, in
/// growing chunks: the nucleus is usually a sliver of the vocabulary, and
/// sorting all 49152 candidates would cost more than everything else in a
/// sample put together.
double keep_nucleus(std::vector<std::int32_t>& ids, const float* weights, double threshold,
                    bool sorted) {
  const Descending order{weights};
  std::size_t ordered = sorted ? ids.size() : 0;  // ids[0, ordered) are final
  std::size_t chunk = 64;
  double kept = 0.0;
  std::size_t n_kept = 0;
  do {
    if (n_kept == ordered) {
      const auto first = ids.begin() + static_cast<std::ptrdiff_t>(ordered);
      ordered = std::min(ids.size(), ordered + chunk);
      const auto last = ids.begin() + static_cast<std::ptrdiff_t>(ordered);
      if (last != ids.end()) std::nth_element(first, last, ids.end(), order);
      std::sort(first, last, order);
      chunk *= 4;
    }
    kept += weights[ids[n_kept++]];
  } while (n_kept < ids.size() && kept < threshold);
  ids.resize(n_kept);
  return kept;
}

/// 53 random bits from one engine output, as a double in [0, 1).
double uniform01(std::mt19937_64& rng) {
  return static_cast<double>(rng() >> 11) * 0x1.0p-53;
}

}  // namespace

void SamplingParams::validate() const {
  // A NaN temperature makes every logit NaN; an infinite one or an infinite
  // penalty can still make one (inf / inf, 0 * inf).
  if (!std::isfinite(temperature)) {
    fail("temperature must be finite, got " + format_float(temperature));
  }
  if (top_k < 0) fail("top_k must be >= 0, got " + std::to_string(top_k));
  // Negated comparison so that NaN is rejected too.
  if (!(top_p > 0.0f)) fail("top_p must be > 0, got " + format_float(top_p));
  if (!std::isfinite(repetition_penalty) || repetition_penalty <= 0.0f) {
    fail("repetition_penalty must be finite and > 0, got " + format_float(repetition_penalty));
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

  // Every logit -inf (or NaN), or some +inf: there is no proper
  // distribution, and the most likely token is the only sensible answer.
  const float max_logit = largest(logits);
  if (std::isinf(max_logit)) {
    if (max_logit < 0.0f) return 0;
    return static_cast<std::int32_t>(std::find(logits.begin(), logits.end(), max_logit) -
                                     logits.begin());
  }

  // 2. Temperature, finite and > 0 here: the constructor validated params_.
  // The logits become (v - max) / temperature, which softmax cannot tell
  // from v / temperature. Dividing v itself by a tiny temperature can push
  // distinct logits to +inf, where they tie and the lowest id wins instead
  // of the largest logit; v - max <= 0, so the quotient can only round down
  // to -inf, probability 0, while the largest logit stays at exactly 0.
  // A NaN logit becomes -inf, an impossible token, so that every ordering
  // below is a strict total order.
  //
  // Top-1 cannot depend on the temperature, but dividing by a huge one can
  // round a logit just below the max to -0, a tie with the max's 0 that the
  // lowest id would win, so top_k = 1 does not divide at all.
  const float temperature = params_.top_k == 1 ? 1.0f : params_.temperature;
  for (float& v : logits) {
    v = (v - max_logit) / temperature;
    if (std::isnan(v)) v = kNegInf;
  }

  float* const x = logits.data();
  const std::size_t n = logits.size();
  const std::size_t k = params_.top_k > 0 && static_cast<std::uint64_t>(params_.top_k) < n
                            ? static_cast<std::size_t>(params_.top_k)
                            : n;
  const bool top_p = params_.top_p < 1.0f;

  // Every token still in play, in index order unless top-k sorts them.
  std::vector<std::int32_t> ids(n);
  std::iota(ids.begin(), ids.end(), 0);

  // 3. Top-k: a selection, not a sort of the whole vocabulary. Sorting the k
  // survivors gives an order that does not depend on how nth_element arranged
  // them, so the sums and the walk below are the same on every platform.
  if (k < n) {
    const Descending order{x};
    std::nth_element(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(k), ids.end(), order);
    ids.resize(k);
    std::sort(ids.begin(), ids.end(), order);
  }

  // Unnormalized softmax weights, written over the logits (they are scratch).
  // The largest logit is now 0, and always among the candidates.
  double total = 0.0;
  for (const std::int32_t id : ids) {
    x[id] = std::exp(x[id]);
    total += x[id];
  }

  // 4. Top-p: the shortest descending prefix holding top_p of the mass. A
  // weight is exp(logit - max), monotonic in the logit, so ordering by weight
  // agrees with ordering by logit except between logits exp rounds together.
  if (top_p) total = keep_nucleus(ids, x, static_cast<double>(params_.top_p) * total, k < n);

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
