# Decision: qualify DF class tuning and reuse final occupied projections

Status: implemented
Date: 2026-09-16

## Problem

Issues #404–407 asked for shared derivative/value autotuning, force-aware
screening and a final-K/force producer-consumer audit. Kernel arithmetic wins
alone had previously failed to improve complete consumers.

## Decision

Share Direct J/K compiler/resource infrastructure, retain real-signature
rankings per workload, and require complete endpoint evidence before promotion.
Select Rys/compact for derivative 000 and polynomial/compact for the other six
classes in the bounded sm_120 384/768 domain. Reuse final physical K's occupied
projection only with full metric rank and an exclusive validated scratch lease;
automatic reuse stays within the existing resident 768-AO admission.

Keep SSS force-budget screening opt-in. Retain generated value candidates and
the batch harness for reproducible investigation, but leave their production
manifest unqualified. The user explicitly authorized closing directions without
speedup rather than continuing to tune indefinitely.

## Evidence

Five interleaved fixed-density, three-update clean runs per policy gave
derivative endpoint medians 0.920860 → 0.907066 s at 384 AO and
4.054524 → 3.980638 s at 768 AO. Final-projection reuse independently gave
4.055600 → 3.970709 s at 768 AO. Projection product count fell from 769 to 3,
and FLOPs from 175154135040 to 61303947264, reusing the existing 754974720-byte U.
384 stayed on fallback. Strict independent force errors remained below 3e-10.
These are individual ablations, not additive speedup claims.

The final combined automatic configuration separately measures
0.924076 → 0.910025 s at 384 AO (1.52%) and 4.059656 → 3.880987 s at 768 AO
(4.40%), again five interleaved samples per arm with three updates throughout.
Final-library regressions pass 144 GPU Python and five GPU native tests.

Screening at 1e-4 skipped 47507808 of 55656960 SSS primitives at 768 AO, but
the complete endpoint changed only from 3.971840 to 3.955689 s. Smaller cases
showed noise or regressions. Four thresholds, 192/384/768 AO and a separately
extended WATER8 fixture are retained. This does not support a universal default.

The first value candidate improved isolated class estimates but regressed the
384-AO cold endpoint (about 10.85 → 12.29 s). Removing evaluator inlining and
restricting cooperative scheduling to measured low classes still gave
10.840220 → 11.600897 s. The revised changed-geometry call was also slower
(9.867199 → 11.800585 s), despite fewer SCF iterations (12 → 9); these branches
are not iteration-matched speedup evidence. No further value promotion was
attempted. The revised smoke has one cold sample per arm, sufficient to reject
promotion but not to estimate a precise regression distribution.

## Rejected alternatives

- Promoting sampled value class scores directly: the native complete export
  pays additional dispatch, state and scheduling costs. Global cooperative
  scheduling also penalizes higher classes with short contractions.
- Recovering raw force projections from truncated metric space: the missing
  directions contribute to the exact spectral Frechet response. Full rank is
  mandatory; a pseudoinverse shortcut is incorrect.
- Retaining a second full U tensor or copying it: existing scratch already
  provides the useful lifetime when its exact producer is validated.
- Automatic force screening based on WATER32: the modest gain does not survive
  all sizes and does not justify a broad accuracy policy.

## Invariants and consequences

Keep FP64, auxiliary basis, metric cutoff and SCF tolerances fixed. Unsupported
Rys derivatives are excluded, not relabeled polynomial. Resource/numerical
failures and profile conflicts cannot select winners. Counters/profiling stay
separate from clean timings. The full-rank proof, scratch invalidation and
unsupported-path fallbacks in [current documentation](../../../../docs/df_tuning.md)
are part of the scientific contract.

The maintained native growth is policy, contraction and scheduling glue; no
second handwritten recurrence was introduced. Diagnostic candidates add code
size, so future expansion needs endpoint justification.

## Revisit when

Value scheduling can keep measured class gains through a complete native
rebuild, or an independently derived higher-class screening bound improves a
held-out endpoint materially. Broaden projection reuse only with equivalent
identity/lifetime and discarded-direction proofs.

## References

- Issues #404, #405, #406, #407.
- [Retained evidence](../../../../benchmarks/results/issue404-407-df/README.md).
- [Prior 000 qualification](2026-09-16-000-rys-qualification.md).
