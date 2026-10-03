# Decision: close CPU KS on the physical maximum commutator

Status: implemented
Date: 2026-10-03

## Problem

A cold ordinary PBE/def2-SVP four-water CPU solve passed its energy, density
change and physical residual RMS gates in 17 iterations / 19 Fock builds, yet
its unchanged snapshot validator rejected the maximum AO commutator entry:
5.422775006813652e-10 against the requested 1e-10 tolerance. RMS was
2.651903838581017e-11. Reconstruction, trace, idempotency and canonicality passed.
The analogous CUDA norm mismatch is recorded in
[the earlier decision](2026-09-23-ks-final-residual-norm.md).

## Decision

Keep the first ordinary RKS iteration trajectory and its two-build physical
finalization. Final acceptance additionally requires the finite maximum
absolute commutator to be <= min(1e-8, density_tolerance), matching the existing
snapshot gate. Existing strict energy, density RMS and residual RMS gates and
all public RMS diagnostics are unchanged.

A rejected ordinary CPU finalization resumes the same factored native RKS loop
from its physically evaluated density and corresponding occupied factor, with
fresh DIIS and only the original unused iteration budget. Fewer than two
remaining iterations cannot establish a new energy difference and fail closed.
Finalization temporaries die before the restart; only terminal success may
retain a physical Fock. True KS component energy, never the generic HF
quadratic energy, remains authoritative.

Incremental RKS keeps its existing accelerator budget plus separately bounded
strict budget and CURRENT-density strict audit. It gains the same maximum gate.
Ordinary first-stage NEXT-density semantics stay distinct. UKS gains only this
gate inside its existing four-correction physical closure, retaining stabilized
occupations. Explicit CUDA COSX host loops retain their prior behavior.

## Work and ownership invariants

For ordinary completed work, I <= max_iterations and Fock builds = I + 2*A,
where A is the number of attempted finalizations. Each physical invocation is
counted separately, including rejected attempts. History is contiguous and has
I entries. No new audit counter, cached canonical frame, seed, public ABI or
export-time solve is introduced. The final-state validator, method-owned leases,
and all force/response acceptance checks remain unchanged. Closing this norm
gap does not imply universal snapshot acceptance on other predicates.

## Evidence and rejected alternatives

The actual-source controller regression fails before this correction and tests
normal/incremental retry, zero/one/two remaining steps, sparse norm separation,
exact counts, inclusive boundaries, nonfinite norm rejection, factor identity,
physical-buffer lifetime, provider failure and unchanged RMS telemetry. Native
controls independently reconstruct physical RKS and UKS energy/residuals,
including stabilized OH and normal/incremental density-factor routes.

Do not loosen snapshot thresholds, substitute maximum values in RMS fields,
restart a full iteration budget, copy an oracle density, fix arrays during
export, or require a blanket unshifted UKS Aufbau projector. The rejected UKS
alternative is recorded in
[the occupation-closure decision](2026-09-22-uks-export-projector-closure.md).
The performance-only PBE mirror change is deliberately separate.

## Restart-baseline diagnostic representation

A restarted local iteration stage begins with no preceding within-stage energy.
The shared solver correctly uses positive infinity for this unavailable energy
change, but the public diagnostic reader previously normalized it only on global
iteration one. A successful ordinary retry therefore produced nonfinite JSON.
This was found and reproduced before the first corrected PBE96 molecular run.

Only the RKS history recorder now emits the reserved finite sentinel -1.0 when
local iteration is one, global offset is positive, and the solver value is the
expected positive infinity. Solver progress, result arithmetic and convergence
gates are untouched. The C descriptor documents this unavailable value; Python
maps exactly a later-row -1.0 to None. Initial positive infinity handling remains
unchanged. This is neither zero energy change nor convergence evidence.

Unlike normalizing every later positive infinity, the explicit marker preserves
unexpected NaN, negative infinity and ordinary-step positive infinity, including
finite-energy subtraction overflow, for fail-closed diagnostics. No public
layout/ABI field or stage counter is added. Actual-source recorder and ctypes
reader tests cover exact stage location, finite measured deltas, unexpected
nonfinites, invalid negative values, and finite JSON after a genuine restart.
