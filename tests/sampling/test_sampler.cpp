// Sampler — the mathematics and the determinism, without the model.
//
// There is no golden data for this module: HF's sampling draws from torch's
// generator, which is not a property worth reproducing. Every case below is
// built so that the right answer is known by construction, and every draw
// uses a fixed seed, so a statistical bound is a deterministic check with
// headroom rather than something that can flake.
//
// ops::argmax is not linkable from here (this test links minillm_sampling
// only), so greedy results are compared against its documented rule instead:
// the first index of the maximum, i.e. std::max_element.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/random.h"
#include "minillm/sampler.h"

using namespace minillm;
using namespace minillm::test;

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr std::int64_t kVocab = 49152;  // SmolLM2's real vocabulary

std::vector<float> random_row(std::int64_t n, std::uint64_t seed) {
  const Tensor t = random_tensor({n}, seed);
  return std::vector<float>(t.f32(), t.f32() + t.numel());
}

/// A row with many exact ties: N(0, 1) samples rounded to multiples of 0.5.
std::vector<float> tied_row(std::int64_t n, std::uint64_t seed) {
  std::vector<float> row = random_row(n, seed);
  for (float& v : row) v = std::round(v * 2.0f) / 2.0f;
  return row;
}

/// ops::argmax's documented rule: the first index of the maximum.
std::int32_t first_argmax(const std::vector<float>& row) {
  return static_cast<std::int32_t>(std::max_element(row.begin(), row.end()) - row.begin());
}

/// sample() treats its logits as scratch, so every call gets a fresh copy.
std::int32_t sample_copy(Sampler& sampler, const std::vector<float>& row,
                         std::span<const std::int32_t> history = {}) {
  std::vector<float> scratch = row;
  return sampler.sample(scratch, history);
}

/// How often each id comes out of `draws` seeded draws from the same row.
std::vector<std::int64_t> draw_counts(const SamplingParams& params, const std::vector<float>& row,
                                      std::int64_t draws,
                                      std::span<const std::int32_t> history = {}) {
  Sampler sampler(params);
  std::vector<std::int64_t> counts(row.size(), 0);
  for (std::int64_t i = 0; i < draws; ++i) ++counts[sample_copy(sampler, row, history)];
  return counts;
}

/// The ids that were drawn at least once.
std::set<std::int32_t> drawn(const std::vector<std::int64_t>& counts) {
  std::set<std::int32_t> ids;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    if (counts[i] > 0) ids.insert(static_cast<std::int32_t>(i));
  }
  return ids;
}

/// Stages 1-4 written the obvious way — a full stable sort, double-precision
/// softmax — returning the ids a correct sampler is allowed to return.
std::set<std::int32_t> reference_support(std::vector<float> logits,
                                         std::span<const std::int32_t> history,
                                         const SamplingParams& p) {
  const std::set<std::int32_t> distinct(history.begin(), history.end());
  for (const std::int32_t id : distinct) {
    float& v = logits[static_cast<std::size_t>(id)];
    v = v > 0.0f ? v / p.repetition_penalty : v * p.repetition_penalty;
  }
  for (float& v : logits) v /= p.temperature;

  std::vector<std::int32_t> ids(logits.size());
  std::iota(ids.begin(), ids.end(), 0);
  std::stable_sort(ids.begin(), ids.end(), [&](std::int32_t a, std::int32_t b) {
    return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)];
  });
  if (p.top_k > 0 && static_cast<std::size_t>(p.top_k) < ids.size()) {
    ids.resize(static_cast<std::size_t>(p.top_k));
  }
  if (p.top_p < 1.0f) {
    const double max_logit = logits[static_cast<std::size_t>(ids[0])];
    double total = 0.0;
    for (const std::int32_t id : ids) {
      total += std::exp(logits[static_cast<std::size_t>(id)] - max_logit);
    }
    double cumulative = 0.0;
    std::size_t keep = 0;
    while (keep < ids.size()) {
      cumulative += std::exp(logits[static_cast<std::size_t>(ids[keep])] - max_logit) / total;
      ++keep;
      if (cumulative >= p.top_p) break;
    }
    ids.resize(keep);
  }
  return {ids.begin(), ids.end()};
}

/// Logits whose softmax at temperature 1 is exactly `probs`.
std::vector<float> logits_for(const std::vector<double>& probs) {
  std::vector<float> row;
  for (const double p : probs) row.push_back(static_cast<float>(std::log(p)));
  return row;
}

SamplingParams with_temperature(float temperature, std::uint64_t seed = 42) {
  SamplingParams p;
  p.temperature = temperature;
  p.seed = seed;
  return p;
}

}  // namespace

// --- parameters -------------------------------------------------------------

TEST_CASE("SamplingParams defaults are greedy") {
  const SamplingParams params;
  CHECK(params.temperature == 0.0f);
  CHECK(params.top_k == 0);
  CHECK(params.top_p == 1.0f);
  CHECK(params.repetition_penalty == 1.0f);
  CHECK(params.seed == 42);
  CHECK_NOTHROW(params.validate());
}

TEST_CASE("SamplingParams::validate rejects out-of-range knobs") {
  const auto rejects = [](auto mutate) {
    SamplingParams p;
    mutate(p);
    CHECK_THROWS_AS(p.validate(), std::invalid_argument);
    CHECK_THROWS_AS(Sampler{p}, std::invalid_argument);
  };
  rejects([](SamplingParams& p) { p.temperature = kNaN; });
  rejects([](SamplingParams& p) { p.temperature = kInf; });
  rejects([](SamplingParams& p) { p.temperature = -kInf; });
  rejects([](SamplingParams& p) { p.top_k = -1; });
  rejects([](SamplingParams& p) { p.top_p = 0.0f; });
  rejects([](SamplingParams& p) { p.top_p = -0.5f; });
  rejects([](SamplingParams& p) { p.top_p = kNaN; });
  rejects([](SamplingParams& p) { p.repetition_penalty = 0.0f; });
  rejects([](SamplingParams& p) { p.repetition_penalty = -1.2f; });
  rejects([](SamplingParams& p) { p.repetition_penalty = kNaN; });
  rejects([](SamplingParams& p) { p.repetition_penalty = kInf; });

  // The disabling values and the far ends of the valid range are all fine.
  SamplingParams p;
  p.temperature = -1.0f;
  p.top_k = 1;
  p.top_p = 1.5f;
  p.repetition_penalty = 0.5f;
  CHECK_NOTHROW(p.validate());
  p.top_p = 1e-9f;
  p.top_k = std::numeric_limits<std::int64_t>::max();
  CHECK_NOTHROW(p.validate());
}

TEST_CASE("non-finite params throw instead of quietly returning token 0") {
  // validate() used to accept each of these, and sample() then returned
  // token 0 where every finite value gives another answer.
  const auto throws = [](const SamplingParams& p, std::vector<float> row,
                         std::vector<std::int32_t> history) {
    CHECK_THROWS_AS(Sampler(p).sample(row, history), std::invalid_argument);
  };
  // NaN temperature: every logit became NaN, then -inf, and the first one
  // won, though token 0 is impossible.
  throws(with_temperature(kNaN), {-kInf, 0.0f, 1.0f}, {});
  // Infinite temperature: inf / inf = NaN took the infinite logit, which
  // otherwise takes all the mass, out of play.
  throws(with_temperature(kInf), {1.0f, kInf}, {});
  // Infinite penalty: the penalized zero became 0 * inf = NaN, which greedy
  // argmax never picks.
  SamplingParams penalized;
  penalized.repetition_penalty = kInf;
  throws(penalized, {-1.0f, 0.0f}, {1});
}

TEST_CASE("SamplingParams::to_string names every knob") {
  SamplingParams p;
  CHECK(p.to_string() == "temperature=0 top_k=0 top_p=1 repetition_penalty=1 seed=42");
  p.temperature = 0.7f;
  p.top_k = 40;
  p.top_p = 0.95f;
  p.repetition_penalty = 1.1f;
  p.seed = 18446744073709551615ULL;
  CHECK(p.to_string() ==
        "temperature=0.7 top_k=40 top_p=0.95 repetition_penalty=1.1 seed=18446744073709551615");
}

TEST_CASE("sample() rejects empty logits and history ids outside the vocabulary") {
  Sampler greedy(SamplingParams{});
  std::vector<float> empty;
  CHECK_THROWS_AS(greedy.sample(empty, {}), std::invalid_argument);

  Sampler warm(with_temperature(1.0f));
  CHECK_THROWS_AS(warm.sample(empty, {}), std::invalid_argument);

  SamplingParams penalized;
  penalized.repetition_penalty = 1.3f;
  Sampler sampler(penalized);
  std::vector<float> row{0.1f, 0.2f, 0.3f, 0.4f};
  CHECK_THROWS_AS(sampler.sample(row, std::vector<std::int32_t>{1, 4}), std::out_of_range);
  CHECK_THROWS_AS(sampler.sample(row, std::vector<std::int32_t>{-1, 2}), std::out_of_range);
  CHECK_NOTHROW(sampler.sample(row, std::vector<std::int32_t>{0, 3}));
}

// --- greedy -----------------------------------------------------------------

TEST_CASE("temperature <= 0 is argmax, ties included") {
  for (const float temperature : {0.0f, -1.0f}) {
    for (std::uint64_t seed = 0; seed < 8; ++seed) {
      for (const std::vector<float>& row : {random_row(1000, seed), tied_row(64, seed)}) {
        // top_k and top_p are ignored by greedy decoding.
        SamplingParams p = with_temperature(temperature, seed);
        p.top_k = 3;
        p.top_p = 0.5f;
        Sampler sampler(p);
        CHECK(sample_copy(sampler, row) == first_argmax(row));
      }
    }
  }
  Sampler sampler(SamplingParams{});
  CHECK(sample_copy(sampler, {7.0f}) == 0);
  CHECK(sample_copy(sampler, {1.0f, 1.0f, 1.0f}) == 0);
  CHECK(sample_copy(sampler, {-3.0f, 2.0f, -1.0f, 2.0f}) == 1);
  CHECK(sample_copy(sampler, random_row(kVocab, 99)) == first_argmax(random_row(kVocab, 99)));
}

TEST_CASE("top_k = 1 is greedy regardless of temperature") {
  // denorm_min and FLT_MAX are the extremes: dividing by the one overflows
  // almost every logit to +-inf, and by the other underflows them to 0.
  const float tiny = std::numeric_limits<float>::denorm_min();
  const float huge = std::numeric_limits<float>::max();
  for (const float temperature : {tiny, 0.01f, 1.0f, 10.0f, 1e4f, huge}) {
    SamplingParams p = with_temperature(temperature);
    p.top_k = 1;
    Sampler sampler(p);
    for (std::uint64_t seed = 0; seed < 8; ++seed) {
      for (const std::vector<float>& row : {random_row(1000, seed), tied_row(64, seed)}) {
        CHECK(sample_copy(sampler, row) == first_argmax(row));
      }
    }
  }
  // Logits around 30 at a subnormal temperature: all three overflow to +inf.
  SamplingParams p = with_temperature(1e-38f);
  p.top_k = 1;
  Sampler sampler(p);
  CHECK(sample_copy(sampler, {29.0f, 30.0f, 28.5f}) == 1);
  // One ulp below the max, divided by FLT_MAX: -2^-24 / FLT_MAX rounds to -0,
  // which ties with the max at 0 unless top-1 skips the division.
  p.temperature = huge;
  Sampler flat(p);
  CHECK(sample_copy(flat, {std::nextafter(1.0f, 0.0f), 1.0f}) == 1);
}

TEST_CASE("as the temperature goes to 0+, every draw is the argmax") {
  // Once (runner-up - max) / temperature is below about -104, exp rounds
  // every weight but the largest to exactly 0, so from that temperature down
  // the draw is the argmax whatever the seed. Each temperature is checked
  // with every filter, many seeds, and several draws per seed.
  constexpr float kFltMin = std::numeric_limits<float>::min();
  std::vector<std::vector<float>> rows;
  for (std::uint64_t seed = 0; seed < 3; ++seed) rows.push_back(random_row(1000, seed));
  // Dividing these by a tiny temperature overflows many logits to +inf.
  rows.push_back({29.0f, 30.0f, 28.5f});
  for (const float scale : {100.0f, 1e30f}) {
    std::vector<float> row = random_row(1000, 10);
    for (float& v : row) v = v * scale + scale / 2;
    rows.push_back(row);
  }
  // All negative, and a runner-up one ulp below the max.
  std::vector<float> close = random_row(1000, 11);
  for (float& v : close) v = -1000.0f - std::abs(v);
  close[500] = -1.0f;
  close[600] = std::nextafter(-1.0f, -2.0f);
  rows.push_back(close);

  std::vector<SamplingParams> configs;
  for (const std::int64_t top_k : {0, 2, 50}) {
    for (const float top_p : {1.0f, 0.9f}) {
      SamplingParams p;
      p.top_k = top_k;
      p.top_p = top_p;
      configs.push_back(p);
    }
  }

  for (const std::vector<float>& row : rows) {
    std::vector<float> sorted = row;
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    REQUIRE(sorted[0] > sorted[1]);  // a unique maximum
    const std::int32_t expected = first_argmax(row);

    std::vector<float> temperatures;
    for (float t = (sorted[0] - sorted[1]) / 128.0f; t > kFltMin; t /= 16.0f) {
      temperatures.push_back(t);
    }
    temperatures.push_back(kFltMin);
    temperatures.push_back(std::numeric_limits<float>::denorm_min());

    for (const float temperature : temperatures) {
      for (SamplingParams p : configs) {
        p.temperature = temperature;
        for (std::uint64_t seed = 0; seed < 8; ++seed) {
          p.seed = seed;
          Sampler sampler(p);
          for (int draw = 0; draw < 4; ++draw) {
            CAPTURE(row.size());
            CAPTURE(temperature);
            CAPTURE(p.to_string());
            REQUIRE(sample_copy(sampler, row) == expected);
          }
        }
      }
    }
  }
}

// --- determinism ------------------------------------------------------------

namespace {

/// A spread of params exercising every stage.
std::vector<SamplingParams> sampling_configs(std::uint64_t seed) {
  SamplingParams plain = with_temperature(1.0f, seed);
  SamplingParams top_k = with_temperature(0.8f, seed);
  top_k.top_k = 20;
  SamplingParams top_p = with_temperature(1.2f, seed);
  top_p.top_p = 0.9f;
  SamplingParams everything = with_temperature(0.9f, seed);
  everything.top_k = 50;
  everything.top_p = 0.95f;
  everything.repetition_penalty = 1.3f;
  return {plain, top_k, top_p, everything};
}

std::vector<std::int32_t> draw_sequence(Sampler& sampler, std::int64_t vocab, int draws) {
  std::vector<std::int32_t> history;
  for (int i = 0; i < draws; ++i) {
    history.push_back(sample_copy(sampler, random_row(vocab, 1000 + static_cast<std::uint64_t>(i)),
                                  history));
  }
  return history;
}

}  // namespace

TEST_CASE("the same seed gives the same sequence across Sampler instances") {
  for (const SamplingParams& p : sampling_configs(123)) {
    Sampler a(p);
    Sampler b(p);
    const std::vector<std::int32_t> first = draw_sequence(a, 256, 100);
    CHECK(draw_sequence(b, 256, 100) == first);
    for (int i = 0; i < 5; ++i) {
      Sampler fresh(p);
      CHECK(draw_sequence(fresh, 256, 100) == first);
    }
  }
}

TEST_CASE("reset() reproduces a sequence of draws exactly") {
  for (const SamplingParams& p : sampling_configs(7)) {
    Sampler sampler(p);
    const std::vector<std::int32_t> first = draw_sequence(sampler, 256, 100);
    sampler.reset();
    CHECK(draw_sequence(sampler, 256, 100) == first);
    sampler.reset();
    CHECK(draw_sequence(sampler, 256, 100) == first);
  }
}

TEST_CASE("a different seed gives a different sequence") {
  const std::vector<SamplingParams> configs_a = sampling_configs(1);
  const std::vector<SamplingParams> configs_b = sampling_configs(2);
  for (std::size_t c = 0; c < configs_a.size(); ++c) {
    Sampler a(configs_a[c]);
    Sampler b(configs_b[c]);
    CHECK(draw_sequence(a, 256, 64) != draw_sequence(b, 256, 64));
  }
}

TEST_CASE("draws are a pure function of mt19937_64's output, on every platform") {
  // With 2^m equal logits every weight is exactly exp(0) = 1 on any libm, so
  // a draw is the top m bits of one engine output — and the standard fixes
  // mt19937_64's output sequence. This pins the whole draw path to something
  // no library or platform can change.
  constexpr int kBits = 10;
  const std::vector<float> uniform(std::size_t{1} << kBits, 0.0f);
  for (const std::uint64_t seed : {0ULL, 42ULL, 0xdeadbeefULL}) {
    for (const float temperature : {0.5f, 1.0f, 3.0f}) {
      Sampler sampler(with_temperature(temperature, seed));
      std::mt19937_64 engine(seed);
      for (int i = 0; i < 1000; ++i) {
        const auto expected = static_cast<std::int32_t>(engine() >> (64 - kBits));
        REQUIRE(sample_copy(sampler, uniform) == expected);
      }
    }
  }

  // The same holds after top-k: the survivors are walked in id order, not in
  // whatever order std::nth_element (which differs between standard
  // libraries) left them. Even ids tie at 0 and survive; odd ids are lower.
  std::vector<float> row = random_row(std::int64_t{2} << kBits, 8);
  for (std::size_t id = 0; id < row.size(); ++id) {
    row[id] = id % 2 == 0 ? 0.0f : -1.0f - std::abs(row[id]);
  }
  SamplingParams p = with_temperature(1.0f, 11);
  p.top_k = std::int64_t{1} << kBits;
  Sampler sampler(p);
  std::mt19937_64 engine(11);
  for (int i = 0; i < 1000; ++i) {
    const auto expected = static_cast<std::int32_t>(2 * (engine() >> (64 - kBits)));
    REQUIRE(sample_copy(sampler, row) == expected);
  }
}

// --- the distribution -------------------------------------------------------

TEST_CASE("a uniform row is covered uniformly") {
  constexpr std::int64_t kTokens = 32;
  constexpr std::int64_t kPerToken = 2000;
  // Each count is Binomial(64000, 1/32): mean 2000, sd ~44. +-400 is ~9 sd.
  const auto check_uniform = [](const std::vector<std::int64_t>& counts) {
    for (const std::int64_t c : counts) {
      CHECK(c > kPerToken - 400);
      CHECK(c < kPerToken + 400);
    }
  };
  check_uniform(draw_counts(with_temperature(1.0f), std::vector<float>(kTokens, 0.0f),
                            kTokens * kPerToken));
  // top_p = 1 with a huge temperature flattens any row towards uniform.
  check_uniform(
      draw_counts(with_temperature(1e6f), random_row(kTokens, 5), kTokens * kPerToken));
  // top_k past the vocabulary and top_p above 1 are both "disabled".
  SamplingParams disabled = with_temperature(1.0f);
  disabled.top_k = 1000;
  disabled.top_p = 1.5f;
  check_uniform(draw_counts(disabled, std::vector<float>(kTokens, 0.0f), kTokens * kPerToken));
}

TEST_CASE("draws follow the softmax probabilities, temperature included") {
  constexpr std::int64_t kDraws = 100000;
  const std::vector<double> probs{0.1, 0.2, 0.3, 0.4};
  const std::vector<float> row = logits_for(probs);

  // Each frequency has sd <= sqrt(0.25 / 1e5) = 0.0016; 0.01 is 6 sd.
  const auto check_frequencies = [&](const std::vector<std::int64_t>& counts,
                                     const std::vector<double>& expected) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
      CHECK(std::abs(static_cast<double>(counts[i]) / kDraws - expected[i]) < 0.01);
    }
  };
  check_frequencies(draw_counts(with_temperature(1.0f), row, kDraws), probs);

  // Temperature 2 takes the square root of each probability before renormalizing.
  std::vector<double> flattened;
  double total = 0.0;
  for (const double p : probs) total += std::sqrt(p);
  for (const double p : probs) flattened.push_back(std::sqrt(p) / total);
  check_frequencies(draw_counts(with_temperature(2.0f), row, kDraws), flattened);
}

// --- top-k and top-p --------------------------------------------------------

TEST_CASE("top-k never returns a token outside the k largest") {
  // Distinct logits in a shuffled order: the k largest are known exactly.
  constexpr std::size_t kTokens = 1000;
  constexpr std::int64_t kK = 10;
  std::vector<std::int32_t> rank(kTokens);
  std::iota(rank.begin(), rank.end(), 0);
  std::shuffle(rank.begin(), rank.end(), std::mt19937_64(3));
  std::vector<float> row(kTokens);
  std::set<std::int32_t> kept;
  for (std::size_t id = 0; id < kTokens; ++id) {
    row[id] = 0.01f * static_cast<float>(rank[id]);
    if (rank[id] >= static_cast<std::int32_t>(kTokens - kK)) kept.insert(static_cast<std::int32_t>(id));
  }

  SamplingParams p = with_temperature(5.0f);
  p.top_k = kK;
  // Every kept token shows up (so this is not secretly greedy) and nothing else does.
  CHECK(drawn(draw_counts(p, row, 5000)) == kept);

  // Ties at the boundary go to the lowest ids, exactly k survive.
  p.top_k = 2;
  CHECK(drawn(draw_counts(p, {3.0f, 2.0f, 2.0f, 2.0f, 1.0f}, 2000)) ==
        std::set<std::int32_t>{0, 1});
}

TEST_CASE("top-p keeps the shortest prefix whose mass reaches top_p") {
  // Four heavy tokens at scattered ids holding 0.95 of the mass, and 96 light
  // ones sharing the remaining 0.05. Cumulative: 0.4, 0.7, 0.9, 0.95.
  std::vector<double> probs(100, 0.05 / 96.0);
  probs[37] = 0.4;
  probs[5] = 0.3;
  probs[81] = 0.2;
  probs[60] = 0.05;
  const std::vector<float> row = logits_for(probs);

  SamplingParams p = with_temperature(1.0f);
  p.top_p = 0.85f;
  CHECK(drawn(draw_counts(p, row, 5000)) == std::set<std::int32_t>{37, 5, 81});
  p.top_p = 0.92f;
  CHECK(drawn(draw_counts(p, row, 5000)) == std::set<std::int32_t>{37, 5, 81, 60});
  // The first token alone already exceeds top_p: it is kept, and alone.
  p.top_p = 0.35f;
  CHECK(drawn(draw_counts(p, row, 2000)) == std::set<std::int32_t>{37});
}

TEST_CASE("top-p always keeps at least one token") {
  SamplingParams p = with_temperature(1.0f);
  p.top_p = 1e-9f;
  // A peaked row and a flat one; on the flat one the tie goes to the lowest id.
  std::vector<float> peaked(50, 0.0f);
  peaked[17] = 20.0f;
  CHECK(drawn(draw_counts(p, peaked, 500)) == std::set<std::int32_t>{17});
  CHECK(drawn(draw_counts(p, std::vector<float>(50, 1.0f), 500)) == std::set<std::int32_t>{0});
  for (std::uint64_t seed = 0; seed < 8; ++seed) {
    const std::vector<float> row = random_row(1000, seed);
    CHECK(drawn(draw_counts(p, row, 20)) == std::set<std::int32_t>{first_argmax(row)});
  }
}

// --- repetition penalty -----------------------------------------------------
//
// Observed through greedy decoding: each case is built so that the argmax
// flips if the penalty is computed wrongly.

TEST_CASE("the repetition penalty divides positive logits and multiplies negative ones") {
  SamplingParams p;
  p.repetition_penalty = 2.0f;
  Sampler sampler(p);
  const std::vector<std::int32_t> history{0};

  // 4 / 2 = 2 < 3. Multiplying would give 8 and keep token 0.
  CHECK(sample_copy(sampler, {4.0f, 3.0f}, history) == 1);
  // -1 * 2 = -2 < -1.5. Dividing would give -0.5 and keep token 0.
  CHECK(sample_copy(sampler, {-1.0f, -1.5f}, history) == 1);
  // Zero stays zero either way.
  CHECK(sample_copy(sampler, {0.0f, -0.1f}, history) == 0);

  // A penalty below 1 encourages repeats: 2 / 0.5 = 4 > 3.
  p.repetition_penalty = 0.5f;
  Sampler encouraging(p);
  CHECK(sample_copy(encouraging, {2.0f, 3.0f}, history) == 0);
}

TEST_CASE("the repetition penalty touches only history ids, each once") {
  SamplingParams p;
  p.repetition_penalty = 2.0f;
  Sampler sampler(p);

  // Only token 1 is penalized: 4 -> 2, and 3.5 wins. Penalizing every token
  // would leave token 1 on top.
  CHECK(sample_copy(sampler, {3.0f, 4.0f, 3.5f}, std::vector<std::int32_t>{1}) == 2);

  // Token 0 appears three times but is divided once: 4 / 1.5 = 2.67 > 2.5.
  // Twice would already give 1.78 and hand the argmax to token 1.
  p.repetition_penalty = 1.5f;
  Sampler once(p);
  CHECK(sample_copy(once, {4.0f, 2.5f}, std::vector<std::int32_t>{0, 0, 0}) == 0);
  CHECK(sample_copy(once, {4.0f, 2.5f, 1.0f}, std::vector<std::int32_t>{2, 0, 2, 0, 2}) == 0);

  // With no history, nothing changes.
  CHECK(sample_copy(once, {4.0f, 2.5f}, {}) == 0);
}

// --- stage order ------------------------------------------------------------
//
// Each case gives a different answer if two adjacent stages are swapped.
// Penalty and temperature are the exception: temperature is a division by a
// positive number, which preserves the sign the penalty branches on, so
// (l / p) / T == (l / T) / p and no input can tell those two orders apart.
// What is observable is that the penalty runs before the filters, and before
// the greedy short-circuit.

TEST_CASE("stage order: the penalty applies under greedy decoding") {
  SamplingParams p;
  p.repetition_penalty = 2.0f;
  Sampler sampler(p);
  CHECK(sample_copy(sampler, {4.0f, 3.0f}, std::vector<std::int32_t>{0}) == 1);
}

TEST_CASE("stage order: the penalty runs before top-k") {
  // Penalized first, token 0 drops to 2.5 and top-1 keeps token 1. With top-k
  // first, token 0 would be the only candidate left and come out every time.
  SamplingParams p = with_temperature(1.0f);
  p.top_k = 1;
  p.repetition_penalty = 2.0f;
  CHECK(drawn(draw_counts(p, {5.0f, 4.9f, 1.0f, 0.5f}, 200, std::vector<std::int32_t>{0})) ==
        std::set<std::int32_t>{1});
}

TEST_CASE("stage order: temperature runs before top-p") {
  // At T = 1 the split is 0.6 / 0.4 and top_p = 0.55 keeps token 0 alone. At
  // T = 4 it flattens to 0.525 / 0.475, so top-p must keep both.
  SamplingParams p = with_temperature(4.0f);
  p.top_p = 0.55f;
  const std::vector<std::int64_t> counts = draw_counts(p, logits_for({0.6, 0.4}), 4000);
  CHECK(counts[1] > 1500);  // expected ~1900
}

TEST_CASE("stage order: top-k runs before top-p") {
  // Full row: 0.3, 0.25, 0.2 and 97 tokens sharing 0.25. After top_k = 2 the
  // survivors renormalize to 0.545 / 0.455, so top_p = 0.5 keeps token 0
  // alone. Top-p on the full row would keep tokens 0 and 1.
  std::vector<double> probs(100, 0.25 / 97.0);
  probs[0] = 0.3;
  probs[1] = 0.25;
  probs[2] = 0.2;
  SamplingParams p = with_temperature(1.0f);
  p.top_k = 2;
  p.top_p = 0.5f;
  CHECK(drawn(draw_counts(p, logits_for(probs), 2000)) == std::set<std::int32_t>{0});
}

// --- the real vocabulary and degenerate rows --------------------------------

TEST_CASE("a [49152] row: every stage, and only allowed tokens") {
  const std::vector<std::int32_t> history = random_ids(64, kVocab, 77);
  for (const SamplingParams& p : sampling_configs(2024)) {
    const std::vector<float> row = random_row(kVocab, 4242);
    const std::set<std::int32_t> allowed = reference_support(row, history, p);
    Sampler sampler(p);
    for (int i = 0; i < 100; ++i) {
      const std::int32_t id = sample_copy(sampler, row, history);
      CHECK(allowed.count(id) == 1);
    }
  }
}

TEST_CASE("NaN and infinite logits never break a draw") {
  SamplingParams p = with_temperature(1.0f);
  // A NaN is an impossible token.
  CHECK(drawn(draw_counts(p, {kNaN, 1.0f, kNaN, 0.5f}, 500)) == std::set<std::int32_t>{1, 3});
  // Nothing is possible: the lowest id, like argmax. A NaN counts as -inf.
  CHECK(drawn(draw_counts(p, std::vector<float>(8, -kInf), 10)) == std::set<std::int32_t>{0});
  Sampler nothing(p);
  CHECK(sample_copy(nothing, {kNaN, -kInf, kNaN}) == 0);
  CHECK(sample_copy(nothing, std::vector<float>(4, kNaN)) == 0);
  // An infinite logit takes all the mass; the first one wins.
  CHECK(drawn(draw_counts(p, {1.0f, kInf, 2.0f, kInf}, 10)) == std::set<std::int32_t>{1});
  // -inf tokens are never drawn, whatever the filters.
  p.top_k = 3;
  p.top_p = 0.99f;
  CHECK(drawn(draw_counts(p, {-kInf, 0.0f, -kInf, 0.0f}, 500)) == std::set<std::int32_t>{1, 3});
}
