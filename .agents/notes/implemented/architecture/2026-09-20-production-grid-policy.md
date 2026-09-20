# Canonical production DFT grid policy (#596)

## Problem

The version-1 reference grid is an exact deterministic fixture contract, but production KS callers also obtained it by default. That duplicated the numerical 48/16/32/3 choices across Python/native entry points and silently supplied a one-Bohr radius to elements with no explicit radius. The reference contract must remain bit-for-bit reproducible while production defaults become element-, method- and accuracy-aware.

## Decision

Version 1 remains the historical reference `GridSpec` with its exact one-Bohr fallback, unpruned topology and existing fixtures. Production resolution is compiler-owned through `GridPolicy/GridProfile` and returns a fully explicit `GridSpec(version=2)`. The modern Python KS path resolves the policy before constructing the native descriptor; C++ consumes only the resolved spec and does not choose a production profile.

Production radii are the pinned `covalent_radius_bohr` values from `external/xtbloom-d3/covalent_radii.json` (SHA-256 `92b32fada844a337204b84f2d961473bad5737240765eb8d0727a62827de5111`, upstream revision `2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`). Version 2 requires a positive radius for every element actually materialized; unsupported elements fail closed instead of becoming one Bohr. Pinned source/upstream provenance is asserted only for a complete v2 spec that exactly equals a canonical `GridPolicy` result. Modified/deserialized custom v2 specs remain valid explicit contracts but are labeled `explicit-grid-v2` with an explicit-radii identity instead of inheriting canonical provenance.

The qualified policy supports LDA and PBE/GGA restricted/unrestricted aliases. Standard LDA and GGA both resolve 54/16/32, retaining the legacy angular workload while adding radial resolution. Tight or first-derivative resolution uses 64/20/40 for LDA and 72/24/48 for GGA. Partition iterations remain three and pruning/screening remain explicitly disabled, so derivative topology is fixed. Meta-GGA/r2SCAN, VV10 and hybrids are outside the qualified boundary and fail closed.

## Rejected alternatives

- Change `GridSpec(version=1)`: would invalidate deterministic reference semantics and fixtures.
- Load native code from the compiler to resolve grids: violates compiler/runtime ownership.
- Duplicate production profile selection in C++: recreates the policy split this issue removes.
- Keep a generic one-Bohr production fallback: hides unsupported elements and lacks provenance.
- Enable pruning/screening while adding the policy: changes derivative topology and expands numerical scope without qualification.

## Invariants

- Version-1 fixture values and semantics are unchanged.
- The resolved version-2 spec is complete identity: version, topology, point counts, partition controls and element radii all participate in KS payload/native snapshot identity.
- Serialization/deserialization round-trips the resolved spec rather than a lossy accuracy label.
- LDA/GGA scientific formulas and XC tolerances are unchanged; only production grid resolution changes.
- Unsupported functional families and unsourced elements fail closed.
- Future consumers such as #175 should call the same compiler policy rather than introducing a second grid policy.

## Evidence

Pure contract tests compare the production radius table exactly with the pinned source, cover light elements, Fe and Xe, verify v1 one-Bohr compatibility, v2 unknown-element failure, LDA/GGA/tight/derivative profiles, capability boundaries, canonical versus explicit-v2 provenance, calculation identity and native descriptor fields. `benchmarks/grid_policy_convergence.py` is the retained independent quadrature gate: PySCF performs its own RKS SCF and analytic grid-response gradient on the resolved VibeQC quadratures; 96×32×64 is admitted only after agreement with 120×40×80, and standard/tight profiles must meet fixed energy/force bounds plus point-count ceilings. The PBE 48×16×32 historical candidate is retained as a negative cost/accuracy control; the promoted 54×16×32 standard profile buys radial accuracy without the prior 18×36 angular-work expansion. Final CPU/CUDA energy/force qualification and exact-head evidence are recorded by the issue gates and PR.
