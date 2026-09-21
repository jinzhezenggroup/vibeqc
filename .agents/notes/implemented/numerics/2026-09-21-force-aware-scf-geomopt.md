# Decision: geometry optimization may adapt SCF effort without owning the optimizer

Status: implemented
Date: 2026-09-21

## Problem

NUM03's geometry-optimization extension needs force-aware SCF effort allocation without
turning an SCF convergence threshold into a universal force-error theorem and without
duplicating the staged/progressive controller owned by #192. VibeQC currently has no
public geometry-optimizer runtime, while #174 already owns mixed-precision safeguards.

## Decision

- Add an opt-in force-aware SCF policy, not a public geometry optimizer.
- The policy changes only SCF convergence controls: energy tolerance, density tolerance,
  and maximum iterations. Grid, screening, finite-basis target and arithmetic policy stay
  under their existing owners.
- Calibrate a limited-domain empirical relation from one explicitly named SCF diagnostic
  to independently measured strict energy/max-force/RMS-force error. The current
  automatic controller uses density RMS because that is the diagnostic directly bounded
  by the public SCF convergence control. Physical residuals may be audited separately but
  are not silently treated as controllable thresholds.
- Calibration identity includes method and a basis-family/representation identity.
  Molecular families are split between training and holdout. An unseen basis may be
  evaluated as research evidence, but production proposals fail closed to strict.
- The first geometry step is strict. Loosening is hysteretic, tightening is immediate,
  and near-stationary steps force strict cleanup. Missing calibration coverage also falls
  back to strict.
- Final acceptance always recomputes energy and forces at the original FP64 strict-SCF
  target and checks the actual optimizer force termination gate.
- A standalone deterministic FIRE harness supplies complete-optimization evidence without
  claiming to be the repository's optimizer API. It caps a single-atom displacement at
  0.20 bohr and removes pure translation.
- The harness compares four complete arms: fixed strict FP64, #174 mixed-arithmetic-only,
  NUM03 tolerance-schedule-only, and the combined policy. Every arm pays for an explicit
  final FP64 strict cleanup. Performance is only comparable when final verification is
  observed-met.

## Rejected alternatives

- Reusing energy convergence alone as a force-error estimate: rejected because force
  sensitivity is the target observable.
- Treating density RMS or physical residual as a rigorous force bound: rejected; the
  relationship is empirical and calibration-limited.
- Automatically extrapolating a STO-3G calibration to def2-SVP: rejected. A held-out
  basis is evidence-only and triggers strict fallback in production proposals.
- Implementing a shared geometry/stage orchestrator here: rejected because #192 owns
  that integration.
- Reimplementing mixed precision inside NUM03: rejected; the benchmark consumes #174's
  existing precision="auto" contract.
- Counting calibration/reference runs as production-policy cost: rejected. They are
  offline research evidence, while runtime retries and final strict cleanup are counted.

## Invariants

1. SCF effort adaptation cannot change the finite-basis model, grid, screening or
   arithmetic ownership.
2. Energy and force allowances are independent.
3. Empirical calibration is never labelled certified or universal.
4. Unknown method/basis domain, missing prior force information and near-stationary
   geometry fail closed to strict SCF effort.
5. Looser nonconverged solves retry at strict effort; failures are not silent success.
6. Every complete optimization ends with an independent FP64 strict-SCF energy+force
   evaluation and the actual force termination gate.
7. Negative complete-endpoint performance remains negative even if an individual loose
   solve is faster.
8. The research harness makes no NVE, Hessian, transition-state or general optimizer
   claim.

## Evidence

Local policy/harness validation on the isolated follow-up worktree:

- force-aware SCF policy plus the existing NUM03 numerical-policy tests: 14 passed;
- Ruff lint and formatting: clean;
- harness import/case construction: clean;
- training families are H2 and LiH; held-out molecular families include water, methane
  and flexible ethane; def2-SVP water is an explicit held-out basis;
- complete benchmark arms include fixed-strict, mixed-arithmetic-only,
  tolerance-schedule-only and combined policies.

qz/Inspire complete-optimization evidence is appended after the exact pushed commit is
run. Compact JSON is retained outside the repository; large raw artifacts are not
committed.

## Consequences

The policy is intentionally conservative. A held-out basis currently receives no SCF
relaxation, and a paired strict cleanup can erase endpoint speedups. This is an acceptable
negative result if it is what matched-final-accuracy evidence shows.

Separate prepared batches retain warm starts within each SCF effort level. Cross-level
state orchestration is not added here; #192 may later provide a shared controller and
checkpoint policy.

## Revisit when

- #192 lands a stable shared staged-controller/optimizer integration seam;
- a proved residual-to-force bound replaces the empirical calibration;
- held-out evidence justifies adding a broader basis-family calibration domain; or
- the repository gains a public geometry optimizer whose exact termination/checkpoint
  semantics should replace the research FIRE harness.

## References

- GitHub issue #175 (NUM03)
- GitHub issue #174 (NUM02)
- GitHub issue #192 (PROG04)
- PR #828 and .agents/notes/implemented/numerics/2026-09-21-force-aware-screening-policy.md
- python/vibeqc/force_aware_scf.py
- tools/vibeqc_numerics/scf_effort_geomopt_benchmark.py
