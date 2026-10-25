// The single place tolerances live.
//
// Do not write a tolerance literal in a test. If a comparison needs a looser
// bound than anything here, that is a conversation about this file (with the
// measured max abs / max rel error in hand), not a local override.
//
// Why anything is loose at all: fp32 addition is not associative, so any
// change in accumulation order — SIMD lanes, thread chunks, tiling, torch's
// pairwise reduction vs our sequential one — moves the last bits. The bound
// that matters is "does the error stay put as inputs grow", not "is it zero".
//
// Every number below is backed by a measurement against oracle/data/ on
// AppleClang 21 / arm64, torch 2.14, recorded in the comment next to it. The
// measured values come from running the suite with all tolerances set to zero
// and reading the reported errors. Redo that if you need to change one.
//
// See docs/ARCHITECTURE.md §5 for the policy.

#ifndef MINILLM_TESTS_COMMON_TOLERANCES_H
#define MINILLM_TESTS_COMMON_TOLERANCES_H

#include <cstdint>

namespace minillm::test {

/// |actual - expected| <= atol + rtol * |expected|
struct Tolerance {
  double rtol;
  double atol;
};

namespace tol {

/// Elementwise, no reduction: add, silu_mul, embedding. The same arithmetic in
/// the same order as torch, so only exp() differs, by a couple of ulps.
/// Measured: abs 3.6e-7, rel 2.2e-7 (silu_mul [8, 1536]).
inline constexpr Tolerance kElementwise{1e-6, 1e-7};

/// One reduction per row: rmsnorm, softmax. torch reduces pairwise, we reduce
/// sequentially, so the error grows with the row length.
/// Measured: abs 2.1e-6, rel 3.0e-6 (softmax over a [1, 49152] logit row);
/// abs 1.9e-6, rel 5.0e-7 (rmsnorm [8, 576]).
inline constexpr Tolerance kReduction{1e-5, 1e-6};

/// matmul: dot products over 576-1536 terms, freely reassociated.
///
/// atol carries this one, not rtol: with N(0, 1) inputs an output element is a
/// sum of k products and lands around sqrt(k) ~ 24-39, but individual elements
/// come out near zero, where the relative error is unbounded and meaningless.
/// atol 2e-4 is ~5e-6 relative to the natural output magnitude, so this still
/// catches a real kernel bug by a wide margin.
/// Measured: abs 6.9e-5, rel 7.5e-3 (lm_head [1, 576] x [4096, 576]) — the
/// large rel is an output element of magnitude ~7e-3. The golden lm_head case
/// trims the vocabulary from 49152 to 4096 rows, so the true max over the full
/// matrix is somewhat higher than measured; hence 3x headroom rather than 1.5x.
inline constexpr Tolerance kMatmul{1e-4, 2e-4};

/// attention: a softmax plus a convex combination of V rows, so outputs stay
/// around the magnitude of V itself (~1 for N(0, 1) inputs) and the error is
/// far smaller than matmul's despite the same number of flops.
/// Measured: abs 6.0e-7, rel 8.6e-5 (q[8, 9, 64], kv[8, 3, 64]);
/// abs 1.2e-6, rel 3.1e-3 on a real layer's post-RoPE q/k/v, where the large
/// rel is an element of magnitude ~4e-4.
/// ops::attention (online softmax, polynomial exp) measured the same way:
/// abs 3.6e-7 on the three oracle cases, abs 1.3e-6 on the real layers, and
/// at most abs 3.3e-6 from ref::attention across tests/attention, that one
/// with logits near 100.
inline constexpr Tolerance kAttention{1e-4, 1e-5};

/// RoPE at small positions. The reference reproduces transformers' inv_freq
/// chain bit for bit, so this is tight. Use rope_tolerance() below whenever
/// positions can be large.
/// Measured: abs 2.4e-7, rel 6.3e-5 at positions 0-7 and at position 4000
/// alike — the error does not grow with position on this platform.
inline constexpr Tolerance kRope{1e-5, 1e-6};

/// Q/K/V and output projections inside an attention block — a matmul on real
/// weights, with room for inputs that already carry error. Real projection
/// outputs reach ~16 in magnitude but most sit well below 1, so unlike
/// kMatmul's N(0, 1) case rtol does real work here.
/// Measured (layers 0 and 15, oracle/data/attention/): abs 2.9e-6, rel 2.0e-3
/// for q/k/v/o_proj each fed its golden input, the large rel coming from
/// elements near zero; abs 1.4e-5, rel 7.8e-3 for the whole block chained
/// from hidden_in. Those are release numbers — debug and asan reproduce the
/// single-stage projections bit for bit. At this bound the worst single
/// stage uses 4% of the tolerance and the chained block 21%, which leaves
/// ~5x for a reassociating matmul.
inline constexpr Tolerance kProjection{1e-4, 1e-5};

/// Per-layer hidden states. 30 layers of residual accumulation; the error at
/// layer i is roughly the error at layer i-1 plus one block's worth.
///
/// The error is set by the first token's row. SmolLM2 parks attention-sink
/// activations there, up to ~2e4 in a handful of channels, so that row's
/// residual sums run at magnitudes where one ulp is ~1e-3. The huge elements
/// themselves stay within 6e-6 relative (rtol covers them 16x over), but every
/// other element of the row inherits an absolute error of a few of those ulps
/// whatever its own size — hence atol does the work, and rtol stays small so
/// the large elements are not waved through.
/// Measured on the model/ prompts (release; asan gives the same numbers):
///   p0,  5 tokens: max abs 2.0e-2 (state 29, an element of 3212), max rel
///                  0.12; worst element uses 12% of this bound (state 11)
///   p1,  6 tokens: max abs 1.2e-2 (state 29), max rel 0.24; 14% (state 10)
///   p2, 15 tokens: max abs 6.8e-3 (state 29), max rel 0.46; 34% (state 29,
///                  an error of 2.1e-3 on an element of 1.5 in row 0)
/// The max rels come from elements near zero. The error does not grow with
/// prompt length: against torch on prefixes of one 444-token text (n = 32,
/// 64, 128, 256) the worst element uses 12-13% at every n, and past row 0 the
/// max abs error per position is 2e-4 to 1.5e-3, flat from position 8 to 255.
inline constexpr Tolerance kHiddenState{1e-4, 6e-3};

/// Final logits (the last row): hidden-state drift plus a [vocab, hidden]
/// matmul. The final norm brings the hidden state back to O(1), so the drift
/// stays small. Greedy token IDS are still compared exactly: a logit tolerance
/// never excuses a divergent token.
/// Measured (release): max abs 4.6e-5 / 3.8e-5 / 3.9e-5 and max rel 5.7e-2 /
/// 1.8e-3 / 3.1e-2 for p0 / p1 / p2, at most 10% of this bound. The same
/// 444-token text at n = 32-256 peaks at 9.3e-5 (22%, n = 128). Prefill then
/// one decode step against a one-shot prefill of the same tokens — the m = 1
/// and m = n matmul paths — differ by at most 5.5e-5 (17%).
inline constexpr Tolerance kLogits{1e-4, 3e-4};

}  // namespace tol

/// RoPE tolerance for data containing positions up to `max_position`.
///
/// The rotation angle is `position * inv_freq[i]` evaluated in float32. Two
/// implementations whose inv_freq differs by one ulp (6e-8 relative — one
/// different libm powf is enough) therefore disagree on the angle by up to
/// `position * 6e-8` radians, and cos/sin are 1-Lipschitz, so that error
/// appears undiminished in the output: 5e-7 at position 8, 5e-4 at position
/// 8192. A single fixed atol would be either too loose to catch a real bug at
/// position 0 or too tight to pass at position 8000.
///
/// minillm's reference reproduces transformers' inv_freq exactly on the
/// platforms we test — the suite passes at the tight kRope bound with this
/// allowance set to zero — so the allowance exists only so that a different
/// math library does not turn into a mystery failure.
inline Tolerance rope_tolerance(std::int64_t max_position) {
  const double angle_slack = 1.5e-7 * static_cast<double>(max_position);
  return Tolerance{tol::kRope.rtol, tol::kRope.atol + angle_slack};
}

}  // namespace minillm::test

#endif  // MINILLM_TESTS_COMMON_TOLERANCES_H
