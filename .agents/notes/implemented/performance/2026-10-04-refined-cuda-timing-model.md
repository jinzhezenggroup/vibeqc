# Decision: refine CUDA timing with constrained family response

Status: implemented
Date: 2026-10-04

## Problem

The [original measured v1 profile](2026-10-03-rtx5090-cuda-timing-calibration.md)
passes its declared accuracy gates, but one occupancy scale cannot describe both
compute and memory saturation. Mixed dependent FMA streams have different achieved
compute rates from independent register chains. Short batches expose a fixed
synchronization/submission intercept, and grids between whole SM waves expose
compute imbalance. Improving accuracy needs new training coverage and fresh
holdouts, not repeated tuning against the original 17 validation cases.

## Decision

Keep v1 calibration serialization and predictions intact. Add explicit
`roofline-calibrated-overlap.v2` profiles with independent monotone memory response,
uniform-block compute wave correction, fixed batch overhead, and bounded
resource-crossover cost. For corrected resource times C and M, body time is
`max(C,M) + k*min(C,M)^2/max(C,M)`, with k in [0,1]. It stays between roofline and
serialized resource time and suppresses crossover cost when one resource dominates.
The no-work body is zero; a known no-op also incurs no fixed overhead.

Use a training-only Theil-Sen launch slope/intercept, geometric-mean achieved
rates, isotonic pooling of memory knots, and a fixed constrained streaming-rate
search. The arithmetic, copy and streaming profiles remain explicitly selected
by callers; GPU identity alone cannot infer dependency structure or precision.
Fitting a family profile does not authorize production default promotion.

Freeze refined median/P95/maximum error gates at 5%/12%/20%, globally and per
held-out family. Training-only reports and incomplete held-out family coverage
cannot qualify. Reject duplicated identities, nonpositive/nonfinite timings,
numerical failures and local-memory probe data. Engineering bands use training
residuals alone, rounded up to 0.01 with a 0.05 floor; coverage is separate from
accuracy and has no confidence interpretation.

## Development and independent evaluation

The old v1 holdouts are now development evidence. Collect an exploratory refined
training-only suite before finalizing the model; it contains no new holdouts.
Then freeze the model and collect 55 training cases plus 27 fresh held-out cases,
nine samples each. New grids, mixed intensities, pure-FMA iteration counts, copy
sizes and launch counts are specified in the probe before the formal fit. Do not
adjust the fit or gates after inspecting their scores.

Compare against a refitted v1 baseline on every same fresh holdout. The baseline
uses its original recipe (pure compute/copy and long launch batches); v2 also uses
mixed training and small launch batches. This measures the complete methods,
not an equal-training formula ablation. All cases and baseline failures remain
in the [retained bundle](../../../../benchmarks/results/cuda-timing-rtx5090-20261004/README.md).
Median/P95/maximum errors improve from 7.39%/23.01%/53.67% to
1.64%/10.36%/14.40%; all refined per-family gates pass. Training-derived bands
cover 22/27 held-out medians, so narrower bands must not be described as reliable
confidence intervals. This single-device suite does not establish performance
outside its FP64 warm serial-batch domain.

## Rejected alternatives

- Continue using one common occupancy scale: it cannot represent distinct memory
  saturation and compute wave imbalance on these training grids.
- Reuse the independent-FMA rate for dependent streaming: the instruction
  structures exhibit different achieved rates, even at matching precision.
- Unbounded polynomial residual fitting: it obscures physical limits and can
  extrapolate negative throughput or implausible overlap. Curves and crossover
  terms have explicit monotonicity and interval constraints instead.
- Always serialize compute and memory: crossover data support intermediate
  overlap, and serialization overpredicts strongly imbalanced work.
- Refit to heldout misses or enlarge the error band to cover them: both leak
  evaluation data. Tests mutate heldout timings and require identical profiles.
- Claim an endpoint improvement: this is an estimator-accuracy change. It has no
  complete chemistry setup, iteration, transfer, force or response measurement.

## Provenance and retention

Slurm job 12183 collected the formal timing run with ccache; job 12186 ran Compute
Sanitizer on that exact binary and reported zero errors. Independent host
std::fma/copy checks satisfy the 2e-12 absolute-error gate. The source hashes in
the measurement resolve exactly to commit `124055b86`, which preserves the dirty
source state used for collection on parent `60e70eff4`. Subsequent docstrings and
conservative evidence checks do not change this run's predictions; CPU replay
checks all profile fields and scored results, including the baseline.

Compress both this bundle and the previous v1 raw JSON losslessly with gzip.
Decoded bytes and SHA-256 identities stay unchanged. The previous PR CI exceeded
the repository's 64 MiB aggregate benchmark budget by about 7 KiB; compressing
task-owned evidence fixes that failure and accommodates the new run without
deleting samples, changing the cap, or publishing external artifacts.

## Consequences and revisit conditions

The compiler remains GPU-free and requires matching explicit calibration. V2
adds workload-specific parameters and assumes uniform block work; it does not
resolve arbitrary instruction dependencies or irregular scheduling. Resource
pressure, spills, cache-resident data, different block sizes/precisions, changed
software/clocks, concurrent streams and chemistry endpoints require independent
training and fresh qualification before broader use. Wider statistical bands
would require repeated independent runs and an explicit coverage target.
