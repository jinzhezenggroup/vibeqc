# Decision: qualify one-step occupied DF warm replay

Status: implemented
Date: 2026-09-26

## Problem

The [shared HF acceptance change](../numerics/2026-09-26-shared-cuda-hf-acceptance.md)
reduced the 768-AO frozen DF replay from five iterations to three, while direct
needed one. DF still discarded its energy baseline, renormalized its returned
density and reconstructed an algebraic seed factor. The seed-to-canonical
energy shift exceeded the common guard even though the density was stationary.

## Decision

Retain a bounded immutable host record only after a strict, exact retained
singleton occupied-RHF final state and all requested endpoint consumers succeed.
The source lifetime token, complete density/Hcore/S/X arrays, occupation and
nuclear energy must match exactly before reuse. The validated D then bypasses
normalization and its canonical occupied C is restored into existing factor
storage. One ordinary seed iteration still builds physical J/K, computes the
pre-DIIS residual, solves the Fock problem, constructs the next D and applies the
unchanged energy/density/residual gates. Subsequent graph iterations remain
available if any gate fails. Final physical validation remains mandatory.

At finalization, J/K are still live under the exact physical occupied-Fock
identity. Reuse them with the existing SCF assembly/reduction kernels to record
the baseline energy. This adds an energy reduction and transfers, not a J/K
evaluation. Copying the public final energy would mix reduction arithmetic and
could reintroduce a spurious first-step delta. The record holds copied occupied
columns, never a lease on response scratch or the next solve's final frames.

Two entries preserve both the latest output and the matched frozen input. Each
new attempted solve first revokes public readiness; a matched immutable entry
survives privately and is republished only with the next successful complete
endpoint. Staging alone and mismatched commit tokens grant no eligibility.
Restore uploads drain before the last pageable snapshot owner is released on
failure. Two records plus temporary full-frame readback are bounded by 64 MiB;
host allocation failure leaves reuse disabled. There is no extra device buffer.

The initial restored factor has generation zero matching the new iteration
counter. Only the ordinary, qualified seed receives this narrow permission;
captured and ordinary subsequent iterations preserve the previous nonzero
generation checks. The first canonical update publishes the normal generation.
Work counters record one warm occupied seed rather than misreporting it as a
dense/algebraic seed or hiding it from the occupied count.

## Boundaries and rejected alternatives

This implements the bounded singleton RHF part of the earlier
[warm-reuse proposal](../../proposed/2026-09-26-df-warm-convergence-parity.md).
UHF, batches, no-DIIS, corrected final states, mismatched inputs/geometry,
missing occupied final J/K, and cache-capacity failures keep their original
bounded path. Changing public tolerances never forces acceptance: the current
iteration still evaluates the requested gates. Geometry changes normally need
multiple iterations before a new same-geometry warm record is established.

No oracle or CPU scientific eigensolve is introduced. Tolerances, the roundoff
factor, final validation and independent output gates are not relaxed. A host
record is preferred to a device copy because it uses existing charged factor
storage on restore and remains independent of force/graph scratch ownership.
`VIBEQC_DF_WARM_REUSE=0` revokes/avoids records for same-binary controls. Restoring
reuse after eviction requires a new accepted output; an old frozen external
input cannot manufacture the lost record from proximity alone.

## Validation and qualification

The native final-snapshot suite includes a closed-form nonidentity-overlap
determinant, one-step iteration-limit replays, missing baselines, exact input
changes, source changes, stale commit generations and failed solve invalidation.
Molecular tests independently compare energies/forces for frozen and advancing
warm state, two auxiliary bases, moved geometry, disabling reuse and requalifying
the returned density. Traces require one actual SCF K and one final occupied K.
Existing RHF/UHF, batch, property-transition, mixed-precision, final correction
and occupied-response regressions remain required.

Complete endpoint measurements retain cold, five original warm, changed geometry,
five moved warm and separate diagnostics. Direct and enabled/disabled DF use the
same new library; each approximation is gated against its independent retained
reference. Same-binary disabled controls use independent owners and seeds, not a
claim of paired identical intermediate densities.

The [retained six-size campaign](../../../../benchmarks/results/df-one-step-warm-20260926/README.md)
contains 184 independently accepted endpoints (168 clean plus 16 diagnostics).
All five original and five moved-warm repeats take one step at 24–768 AOs.
At 768 AOs, enabled DF is 6.174827 s versus 8.021126 s with reuse disabled in the
same binary and 3.048253 s for direct: 23.0% less DF endpoint time, still 2.03×
direct. The control owners establish their own frozen seeds. Cold and changed
geometry still require normal iteration. One-step DF traces show one SCF K plus
one final K and zero additional J/K builds for baseline retention. This remaining
endpoint gap cannot be attributed to extra warm SCF iterations.

Four native suites, 44 existing GPU cases and four new molecular cases pass.
The native final-snapshot/cache suite has zero Compute Sanitizer memcheck errors;
134 host protocol/structure/ownership tests pass. The evidence receipt preserves
the initial native fixture reservation error, moved-geometry test error and
sanitizer PATH retry, alongside the passing corrections. None relaxed production
acceptance. Independent all-endpoint errors are at most 2.6421e-10 Eh and
1.4052e-10 Eh/Bohr under unchanged 1e-8 / 1e-7 gates.

## Revisit when

A device-resident implementation can preserve the frozen and advancing records
with explicit resource charging and measured endpoint gains, or UHF/batch and
corrected-state reuse can be independently qualified under the same contract.
