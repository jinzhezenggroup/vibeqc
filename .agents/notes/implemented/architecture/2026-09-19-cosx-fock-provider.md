# Decision: Register COSX as an explicit prepared exchange provider

Status: implemented
Date: 2026-09-19

## Problem

The native CUDA COSX candidate had correct bounded device algebra but lived
outside the common #202 Fock lifecycle. Simply adding an enum would not be
sufficient: the discrete COSX grid must be part of mathematical identity, the
point tile must be part of execution identity, direct/DF Coulomb providers must
remain independently composable, and COSX memory must be admitted before other
providers consume the device budget.

## Decision

Register `SeminumericalCosx` as an exchange-only CUDA provider with maximum
derivative order zero. `FockCosxSpec` is independent of the XC GridSpec type
and records the v1 quadrature/symmetrization/fitting/screening semantics.
`cosx_tile_points` is stored separately in `ResolvedFockBuild`.

`PreparedFockPlan` converts the resolved COSX prescription only at execution,
materializes the explicit grid, reserves the exact bounded COSX device footprint
first, then gives the remaining budget to the unchanged direct/DF planners.
The prepared owner holds the COSX grid and CUDA plan for its full lifetime.
`CudaFockProviderView` returns raw positive K matrices for RHF or independent
UHF spin densities; the common Fock composition retains coefficient ownership.

CUDA registration is executable only for explicit requests. No AUTO policy is
changed, batching remains unadvertised, and derivatives are rejected.

## Invariants

- COSX can provide exchange only; J remains exact or density-fitted.
- Mathematical grid/approximation changes invalidate provider identity.
- Point-tile changes invalidate prepared execution identity.
- COSX device bytes are reserved before direct/DF allocation and total retained
  provider bytes may not exceed the caller budget.
- The CPU COSX implementation remains an independent oracle, not a production
  provider.
- AUTO selection, analytic forces, screening/fitting, and performance promotion
  remain unsupported until separately evidenced.

## Evidence

`test_fock_build.cpp` gates the exchange-only capability and identity rules.
`test_cuda_fock_composition.cpp` compares direct-J/COSX-K and RI-J/COSX-K
against independent direct, DF, and discrete COSX CPU oracles for RHF and UHF,
and checks cache invalidation, exact resource admission, and derivative
rejection.

## Consequences

The provider is usable by explicit prepared-Fock consumers and can be exercised
inside SCF, but it is not yet an automatic or performance-qualified route.
The correctness-oriented ESP kernel still lacks chain-of-spheres screening and
optimized local task schedules.

## Revisit when

Large-system/TZ/QZ benchmarks establish a reproducible crossover against the
qualified occupied RI-K baseline, or when complete COSX derivatives are
implemented and validated against the same discrete model.

## References

- #246
- #202
- docs/cosx_reference.md
- tests/native/test_cuda_fock_composition.cpp

Agent: ChatGPT
Model: GPT-5.6 Sol
