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

The bounded reuse policy applies to singleton RHF.
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

## Historical five-step diagnosis

The following diagnosis and proposal describe the earlier solver at commit
`77a97a07c9687387d82baf4932e7654c18798e95`, before the decisions above were
implemented. They are retained here to preserve the measured failed shortcuts.

### Original problem


The [final occupied-K qualification](../../../../benchmarks/results/df-final-occupied-endpoint-20260926/README.md)
reports five DF SCF iterations versus one direct iteration on the frozen
768-AO water32mer seed. Equal public tolerances do not imply equal effective
stopping policies. These are complete endpoint measurements, not equal-work
measurements of the two J/K builders or evidence of an intrinsic fivefold
difference in convergence.

### Confirmed behavior

`src/scf/cuda/df_rhf_scf.cpp` initializes the previous energy to infinity on
every execution, including same-geometry warm replay. Its convergence kernel
therefore cannot accept the first iteration. It also resets DIIS and reconstructs
the initial exchange factor from the uploaded density.

`src/scf/cuda_rhf.cpp` retains an energy baseline bound to the exact warm density,
geometry and plan. Frozen replay restores its corresponding baseline. Its
convergence kernel additionally permits a direct-Fock FP64 comparison guard
of `16 * epsilon * max(1, abs(E), abs(previous_E))`, to account for atomic
reduction order. At this geometry that guard is about `8.63748e-12 Eh`, on top
of the requested `1e-12 Eh`. Direct also requires its physical residual gate;
the guard is not permission to accept an unconverged density.

Observation-only kernel prints reproduce the original five DF iterations:

| DF iteration | Absolute energy change (Eh) | Density-step RMS | Reason to continue |
|---:|---:|---:|---|
| 1 | infinity | 2.060875e-13 | Missing energy baseline |
| 2 | 2.009983e-10 | 8.557401e-14 | Energy change |
| 3 | 4.547474e-12 | 6.564152e-14 | Energy change |
| 4 | 2.728484e-12 | 6.601146e-15 | Energy change |
| 5 | 9.094947e-13 | 4.783882e-15 | Accepted |

The density criterion (`1e-10`) passes in every row. The energy criterion alone
blocks iterations 2 through 4. The direct diagnostic replay accepts one step
with energy change `1.818989e-12 Eh` and density RMS `8.069820e-13`; its energy
change would fail the unguarded DF comparison. The initial seed-to-canonical
energy shift and the later few-ULP changes should not be conflated: the former
is roughly `2e-10 Eh`, larger than direct's guard.

The DF warm seed is the density returned by strict final-state validation:
`finalize_density_fitting_rhf` assigns the selected density to the result before
`FleetPlan::retain_warm_state` stores it. This is not a cache of the pre-validation
density.

### Discriminating controls

On one frozen DF seed, a diagnostic-only early return after symmetrization
preserved its values instead of applying the warm electron-trace rescaling.
The original trace was `320.00000000000563`, giving scale
`0.99999999999998246`. This bypass was confined to the diagnostic library; it
is not an admitted production policy for external or changed-geometry seeds.

| Normalization | First-step K | SCF iterations | Second energy change (Eh) |
|---|---|---:|---:|
| Existing | Factor | 5 | 2.009983e-10 |
| Existing | Dense | 4 | 2.673914e-10 |
| Preserve values, diagnostic only | Factor | 3 | 2.310117e-10 |
| Preserve values, diagnostic only | Dense | 3 | 2.946763e-10 |

Neither preserving the seed nor selecting dense seed K removes the second-step
shift. Thus neither normalization alone nor algebraic seed K alone explains it.
The exact division of that shift among seed/factor/projector arithmetic remains
unresolved. Its sensitivity and the later ULP-sized differences show why changing
one rounding path can change the observed iteration count without a meaningful
density improvement. No speedup or complete fix is claimed for these controls.

### Proposed direction and invariants

Retain a validated same-geometry density/occupied-frame state and an energy
baseline with compatible arithmetic and exact model/lifetime provenance. A
cached host final energy cannot simply be copied into the device comparison:
the final and iterative energy reductions and factor paths differ. Preserve
normalization and bounded fallbacks for imported or changed-geometry inputs.

Qualify the effective energy and physical-residual contract against an
independent oracle before selecting any DF roundoff treatment. Do not copy
direct's guard mechanically, remove strict final validation, or force a desired
iteration count. Re-measure complete cold, warm and moved endpoints, keeping
actual work counts and unchanged energy/force acceptance gates.

### Evidence

Source `77a97a07c9687387d82baf4932e7654c18798e95`; RTX 5090, Slurm `main`, CUDA
12.9.1, eight host threads, identical molecular protocol and references to the
linked qualification. Job 11800 measured six DF and four direct endpoints;
job 11801 measured one DF cold endpoint and four frozen-seed controls. All 15
pass independent `1e-8 Eh` / `1e-7 Eh/Bohr` gates; maximum observed errors are
`2.6511771e-10 Eh` and `1.4179769e-10 Eh/Bohr`. Diagnostic times are not added
to the clean benchmark medians.

Ignored raw evidence is in `.artifacts/df-warm-iteration-diagnosis/`, including
copied instrumented sources, exact compile/link argument arrays, runners,
per-endpoint JSON, and logs. Production sources and the original measured
library were untouched. The first diagnostic library only adds convergence
prints; the second adds normalization observations and the explicit bypass.

| Artifact | SHA-256 |
|---|---|
| Observation library | `d70dc7bc4ab47089482d554268af7ec19889f11ef26857940b5abdfb573c0317` |
| Normalization-control library | `ab727c1aedc5d28e02093406b1e4a12fdb226b82412afc02d547e816fe4944c6` |
| Job 11800 log | `1b4c1c5efd587fc3732dc444b5ca369b63d2d872b8c7a90b6679c9278a0df2d5` |
| Job 11801 log | `7a3bc7bf01b9e584b32a60ba5d7a0dc9db678e1f3961a87e05aa8efa9ca41256` |
