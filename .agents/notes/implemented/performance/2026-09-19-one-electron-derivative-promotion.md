# Decision: promote generated one-electron gradients and retire the scalar native force

Status: implemented
Date: 2026-09-19

## Problem

CG04/#141 delivered compiler-owned S/T/V first derivatives and a fused weighted
CUDA consumer, but production still defaulted to native derivative/force code.
That left duplicate scientific implementations in the normal path even after
the generated shell-warp schedule passed strict numerical/resource validation.

## Decision

Generated one-electron derivatives are the CUDA production default, with
`shell_warp` as the default schedule. `VIBEQC_ONE_ELECTRON_DERIVATIVES=reference`
selects the retained cooperative native route for comparison and documented
performance exceptions. The slower scalar native force kernel, launch wrapper,
policy switch, and `VIBEQC_ONE_ELECTRON_FORCE_SCALAR` runtime identity are
removed.

The retained native exception is not a silent fallback: generated launch
failures remain failures. CPU/libcint validation remains structurally
independent of the generated CUDA DAG.

## Rejected alternatives

Deleting every native derivative path was rejected because the archived #141
endpoint campaign contains real counterexamples: resident DF unchanged replay
was faster on the retained route, and Direct sdf18 changed geometry also favored
the retained route. Keeping generated derivatives opt-in was rejected because
the qualified generated shell-warp consumer is the stronger normal ownership
boundary and avoids permanent duplicate formula ownership. Automatic
shape-specific switching was deferred until a stable predictor is independently
qualified.

## Invariants

- Stationary HF supplies density for T/V and negative energy-weighted density
  for S; the generated consumer does not differentiate those weights.
- Nuclear repulsion remains a separate physical term and is accumulated once.
- RHF/UHF spin factors, failed-item masks, Cartesian/spherical conventions and
  changed-geometry invalidation remain unchanged.
- `reference` is explicit; CUDA failures do not silently select native math.
- Independent CPU/libcint raw-value and finite-difference validation remains.

## Evidence

The pinned RTX 5090 #141 campaign reports maximum raw derivative error
`6.67e-14` and arbitrary-weight contraction error `1.07e-13`. On sp8 batch 3,
mean one-electron kernel time was 335.6 us for scalar native, 85.2 us for
cooperative native, 121.8 us for generated AO threads and 58.0 us for generated
shell warp. Complete force comparisons passed their numerical gates.

The retained exception is justified by archived endpoint medians: resident DF
unchanged replay was 4.586 / 5.429 ms (reference / generated), while Direct
sdf18 changed geometry was 36.667 / 39.352 ms. No new GPU speed claim is made by
this retirement change; these pinned measurements define its current boundary.

## Consequences

Normal CUDA one-electron derivative science now has one compiler-owned formula
source. The explicit cooperative exception still carries maintenance cost, but
the scalar family and its policy surface are gone. Benchmark scripts reject
scalar mode on current source and direct users to an archived pre-#357 checkout.

## Revisit when

Re-run the retained resident-DF and sdf18 counterexamples after schedule/runtime
changes. If generated execution no longer regresses either endpoint within the
declared gate, delete the cooperative exception and its remaining numerical
support.

## References

- #141 generated one-electron derivatives
- #357 derivative/force retirement
- `docs/one_electron_derivatives.md`
- `benchmarks/results/one-electron-derivatives-rtx5090/`

---
Agent: ChatGPT
Model: GPT-5.6 Sol
