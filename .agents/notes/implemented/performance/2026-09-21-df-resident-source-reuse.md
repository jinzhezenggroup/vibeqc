# Decision: preserve resident DF value reuse across response panel widths

Status: implemented
Date: 2026-09-21

## Problem

PR #614 resolves a zero public DF budget to positive value/response allowances.
That routes ordinary dense HF preparation through the generated source owner.
The old host-raw J/K-scratch response remains intentionally ineligible for that
layout. However, the bridge already has an identity-checked, immutable whitened
forward tensor and used it only when the response panel was narrower than Naux.
A full-width panel discarded that reader and regenerated raw integrals. The
source guard also excluded its qualified shell consumer and BLAS default.
Separately, the probe-failure 1-GiB ceiling was applied even to roomy live GPUs.

## Decision

- Let a validated full-rank whitened owner select the existing architecture/work
  qualified shell response; explicit diagnostic controls still take precedence.
- Use the existing fitted reader for full-width as well as partial panels, and
  dispatch that reader before the raw full-width contraction. This eliminates
  raw regeneration without a new persistent tensor or scratch alias.
- Keep the 32-MiB to 1-GiB deterministic envelope when probing is unavailable.
  With a live probe, bound the workload target by actual free memory after the
  existing reservations. Keep every positive user budget unchanged. Increment
  the resource-policy version because automatic resolved capacities can differ.

## Invariants

Full-rank/metric/source identities and alias checks are unchanged. A truncated
metric never borrows the full-rank fitted route. Borrowed J/K and packed occupied
paths retain their existing capacity and lifetime guards. This does not promote
packed response automatically, enable screening, relax convergence/error gates,
change precision, remove force final-state validation, or call a CPU oracle in
production. GPU source generation remains the bounded fallback when no forward
value owner fits. Response scratch remains separately charged.

## Rejected alternatives

Removing the J/K-scratch source guard would lend buffers whose actual capacities
can be only a small tile. Reverting resource accounting or widening explicit
budgets would violate bounded execution. A hardware/model-size special case is
unnecessary: the existing validated view and live resource envelope suffice.

## Evidence

The new live-budget regression fails on the parent: its 768-by-768 dense tensor
cannot fit under the 1-GiB ceiling. The independent 96-AO PySCF energy/force gate
passes numerically on the old binary but the new work assertion fails with
`raw_tile_productions=96` instead of zero. GPU regression coverage includes RHF,
UHF, full/partial panels, changed/restored geometry, and energy/force replay.
Final build, numerical and complete-endpoint evidence is recorded in the PR.

## Revisit when

A future low-memory/cost planner changes which immutable forward owner fits, or
rank-truncated response receives a separately validated fitted representation.
A different reduction/order must repeat the independent complete-force gates.

## References

#614, #598, #443, #206. This supplements the 2026-09-20 DF resource-policy note.

Agent: ChatGPT
Model: GPT-6 Astra Pro
