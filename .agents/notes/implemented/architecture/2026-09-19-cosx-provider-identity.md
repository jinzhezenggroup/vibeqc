# Decision: Keep COSX approximation identity explicit before execution promotion

Status: implemented
Date: 2026-09-19

## Problem

The #202 Fock abstraction originally distinguished exact and density-fitted
providers, but COSX is a separate seminumerical approximation whose grid,
symmetrization, fitting and screening choices change the mathematical result.
Treating COSX as a CUDA schedule of exact or RI-K would let cache identity,
derivative dispatch or AUTO selection silently change the Hamiltonian.

The bounded CUDA candidate from #246 is numerically qualified, but it is not
yet owned by PreparedFockPlan and has no analytic derivative path.

## Decision

Add SeminumericalCosx as a distinct internal Fock approximation. Its term
identity includes FockCosxSpec: COSX version, grid version, radial/angular
sizes, partition iterations, coincident-center tolerance, all element radii,
symmetrization, overlap fitting and screening.

COSX v1 is full-range exchange only. Its provider domain supports restricted
and unrestricted spin densities, Cartesian and spherical AOs through f, and
arbitrary exchange coefficients, but advertises no Coulomb, batching or
derivative capability. Version 1 requires explicit symmetrization and rejects
overlap fitting and screening.

The CUDA registration remains Reserved after numerical qualification. The CPU
path remains an oracle and is not registered as executable. Semantic
resolution may represent RI-J + COSX-K, but execution preflight continues to
fail until PreparedFockPlan owns the COSX source and resources.

The public C Fock ABI remains exact/DF only for this slice. There is therefore
no public setting that can silently select COSX.

## Rejected alternatives

- Reuse Exact and add a COSX schedule: rejected because grid/model error would
  disappear from mathematical identity.
- Reuse DensityFitted because both are approximate exchange: rejected because
  auxiliary-basis/metric semantics and seminumerical grid semantics are
  unrelated.
- Mark the CUDA candidate executable immediately: rejected because provider
  lifetime/resource ownership, batching and derivative consistency are not yet
  implemented.
- Put COSX grid data in XC GridSpec implicitly: rejected because COSX
  quadrature is independently versioned and may diverge from XC quadrature.

## Invariants

- COSX cannot be resolved as a Coulomb provider.
- COSX v1 cannot inherit derivative order one from exact/DF providers.
- Any change to the explicit COSX grid/model fields changes resolved identity.
- Irrelevant COSX metadata is canonicalized away for exact/DF terms.
- AUTO or provider availability cannot turn exact/DF exchange into COSX.
- Execution registration must remain non-executable until the prepared owner
  can reproduce the qualified bounded CUDA path.

## Evidence

vibeqc_fock_build_tests checks RI-J/COSX-K resolution, exchange-only and
energy-only capability boundaries, grid-identity invalidation and explicit
rejection of missing identity, fitting, screening, invalid radii and COSX-as-J.

The native CUDA arithmetic remains independently qualified by
vibeqc_cosx_cuda_tests against the Slice-A CPU discrete oracle.

## Consequences

MethodIR and later hybrid-DFT adapters can now name COSX without special-casing
it as a backend optimization. A subsequent slice can wire the reserved CUDA
registration into PreparedFockPlan without changing the mathematical schema.

## Revisit when

PreparedFockPlan owns the COSX grid/source/resource lifetime and fixed-density
RHF/UHF provider tests pass. At that point the CUDA registration may become
Executable for derivative order zero. Derivative capability requires a
separate promotion after the complete discrete COSX gradient is implemented.

## References

- #202
- #246
- PR #572
- docs/fock_build.md
- docs/cosx_reference.md

Agent: ChatGPT
Model: GPT-5.6 Sol
