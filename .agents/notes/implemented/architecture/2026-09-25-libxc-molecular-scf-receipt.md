# Decision: require dual-spin lifecycle evidence for bulk Libxc molecular SCF

Status: implemented
Date: 2026-09-25

## Problem

The generic RKS/UKS consumer boundary exists, but the capability DAG still needs
a precise producer contract for molecular-SCF evidence. A single converged energy
or a one-spin result is insufficient to demonstrate the automatic path expected
by #1121, and a stage-level pass must not erase the exact compiled artifact and
KS plan that were executed.

## Decision

Define a versioned molecular-SCF receipt that consumes the exact
BulkKsResolution v2 objects produced after compiled-CPU and production-domain
qualification.

A successful receipt requires both spin layouts and three lifecycle phases per
layout:

- cold;
- warm replay on the same fixture and geometry; and
- changed geometry on the same fixture but a distinct geometry identity.

Every passing row records a fixture, geometry and independent-reference identity,
convergence, positive iteration count, total energy, independent reference
energy, self-consistent absolute energy error/tolerance, and physical residual
with its tolerance.

The receipt retains each spin's detached KS resolution and hashes it. Those
resolutions already carry the exact compiled point-binding/result identities.

Only a complete all-pass receipt emits molecular-scf stage evidence. Its endpoint
qualification covers CPU energy for both polarized and unpolarized layouts using
the existing endpoint-coverage schema. Fail/not-run rows remain non-promoting and
keep an actionable reason.

## Rejected alternatives

- Promoting from one successful RKS calculation would not qualify UKS.
- Treating a warm replay as a second independent geometry would hide invalidation
  errors.
- Reusing the cold geometry for the changed-geometry row would not exercise
  rebuild semantics.
- Recording only an energy without an independent reference and residual would
  turn convergence into a self-reference.
- Reconstructing a KS plan from the functional name when writing the receipt
  would discard the exact compiled execution provenance from #1338.

## Invariants

- Molecular-SCF evidence is downstream of exact compiled-CPU and
  production-domain evidence.
- Both spin layouts are required for automatic stage promotion.
- Cold/warm/changed lifecycle semantics are explicit and identity-bound.
- Independent energy and physical residual gates must both pass.
- This receipt grants energy endpoint coverage only; forces/response remain
  separate stages.
- No public-method capability follows automatically.

## Evidence

Focused tests cover the complete dual-spin matrix, missing rows, warm geometry
drift, changed-geometry reuse, inconsistent independent energy errors, residual
failures, non-promoting failure rows, and stored KS-resolution tampering.

Repository CI is the executable validation authority for this stacked slice.

## Consequences

The next implementation slice can focus narrowly on executing the existing
generic native RKS/UKS consumer and writing these rows. It cannot weaken the
acceptance definition inside its runner.

## Revisit when

The automatic product contract intentionally changes its required lifecycle
phases or endpoint coverage, or a future common molecular evidence schema
supersedes this Libxc-specific receipt adapter.

## References

- #1118
- #1121
- #1234
- #1338

Agent: ChatGPT
Model: GPT-5.6 Sol
