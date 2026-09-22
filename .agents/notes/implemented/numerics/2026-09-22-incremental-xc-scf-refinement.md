# Decision: Exact incremental XC is an accelerator, never the final SCF authority

Status: implemented
Date: 2026-09-22

## Problem

Issue #237 needs to reuse XC work between SCF iterates without treating an
indefinite delta density as a physical density and without allowing cached state
to survive a change of geometry, basis, grid, functional or numerical policy.
The fixed-model Slice-A primitive from #938 establishes exact nonlinear PBE
differences, but promoting that primitive into SCF adds state-lifetime,
convergence and resource-accounting hazards that fixed-density parity alone does
not cover.

## Decision

The first native SCF integration is an opt-in CPU PBE-RKS correctness baseline.
It uses an anchor-plus-total-delta representation: the accepted full anchor
density stays fixed until a transactional rebuild, and every incremental call
forms the total density features before the existing nonlinear PBE evaluator.

Each solve mints a fresh immutable model identity after the ordinary prepared
geometry/basis/grid validation. The identity includes geometry and basis
fingerprints, the full GridSpec, functional scales/domain policy, screening,
requested/effective precision, XC execution mode and density-source kind. Each
successful anchor replacement receives a monotonically increasing source
generation. Same-shaped data is therefore never sufficient to authorize reuse.

Full rebuilds are available for update-count, anchor-drift, near-cancellation
noise, stagnation and invalid-feature fallback. Replacement is transactional:
the full evaluation and replacement density are complete before the previous
anchor is swapped out. Diagnostics account for retained anchor bytes, delta
buffers and old+replacement overlap.

Incremental XC never owns final convergence. After the incremental stage ends,
the driver clears DIIS, drops the anchor and runs the same native SCF driver in a
strict full-XC phase with its own bounded iteration budget. A successful strict
phase is followed by an independent full requested-target build and physical
commutator-residual audit at the exact candidate density. Audit failure clears
DIIS and continues full iterations while strict budget remains. No incremental
state is used for forces.

## Rejected alternatives

- A single post-SCF full Fock/XC build is not sufficient: it checks an operator,
  not density convergence under that operator.
- Carrying an anchor across prepared replays was rejected because shape equality
  cannot prove geometry/grid/basis/functional identity.
- Treating `E_xc[delta-D]` or `V_xc[delta-D]` as an increment remains invalid;
  delta-D participates only in linear feature contraction.
- Approximate local skipping is not part of this exact mode. It requires a
  separately identified #175/#234 error policy and its own evidence.

## Invariants

- The default/full PBE path is unchanged; incremental XC is default-off.
- Only total rho/grad-rho enter nonlinear PBE; sigma cross terms are retained.
- Failed full rebuilds cannot replace an accepted anchor.
- Geometry/basis/grid/functional/screening/precision/math/source changes create
  a new model identity rather than reusing an anchor.
- Final success always comes from strict full-XC iterations plus an independent
  physical residual audit.
- Force evaluation remains on the independently qualified full stationary
  derivative path.

## Evidence

`vibeqc_dft_density_source_tests` covers full/incremental SCF parity, periodic and
drift/noise rebuilds, same-shaped changed-geometry identity, warm replay,
short-budget failure isolation, strict refinement/final audit execution and
resource counters. Fixed-density cancellation/indefinite-delta and nonlinear
feature parity remain covered by the #938 tests.

## Consequences

This exact baseline may be slower than a full build because the Slice-A
primitive recomputes anchor and delta features. That is an accepted correctness
baseline, not a performance claim. Production promotion requires matched
complete-solve evidence and, for any local skipping, a separately validated
error controller.

## Revisit when

Revisit the retained representation when a qualified implementation can retain
bounded anchor grid features or safely skip local work with #175/#234 error
evidence and demonstrates complete endpoint savings after anchor preparation,
rebuilds, retries, strict cleanup and memory overhead are included.

## References

- Issue #237
- PR #938 (exact fixed-model Slice A)
- Issues #173, #175, #203, #234 and #459
