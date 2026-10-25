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
inline constexpr Tolerance kAttention{1e-4, 1e-5};

/// RoPE at small positions. The reference reproduces transformers' inv_freq
/// chain bit for bit, so this is tight. Use rope_tolerance() below whenever
/// positions can be large.
/// Measured: abs 2.4e-7, rel 6.3e-5 at positions 0-7 and at position 4000
/// alike — the error does not grow with position on this platform.
inline constexpr Tolerance kRope{1e-5, 1e-6};

/// Q/K/V and output projections inside an attention block — a matmul on real
/// weights, so matmul-like bounds with room for inputs that already carry
/// error.
/// PROVISIONAL: nothing measures this until workflow 06 compares against
/// oracle/data/attention/*_q_proj.npy and friends. Replace with measured
/// values then.
inline constexpr Tolerance kProjection{2e-4, 2e-4};

/// Per-layer hidden states. 30 layers of residual accumulation; the error at
/// layer i is roughly the error at layer i-1 plus one block's worth.
///
/// PROVISIONAL: a placeholder until Model::forward exists and the real drift
/// is measured. The workflow that lands the forward pass must replace this
/// with measured values and record the measurement in its PR.
inline constexpr Tolerance kHiddenState{2e-3, 2e-3};

/// Final logits: hidden-state drift plus a [vocab, hidden] matmul.
/// PROVISIONAL, as above. Note that greedy token IDS are compared exactly —
/// a loose logit tolerance never excuses a divergent token.
inline constexpr Tolerance kLogits{1e-3, 5e-3};

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
