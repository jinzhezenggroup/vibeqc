# Decision: compose generated Coulomb J with independent exact K

Status: implemented
Date: 2026-09-25

## Problem

The resident CUDA direct provider already owns a generated pure-Coulomb consumer,
but it previously used that consumer only for J-only requests. A global-hybrid
value request contains both J and exact K, so the provider discarded the
generated-J opportunity and sent both terms through the generic public-AO
independent J/K kernel. That reintroduced the generic Coulomb traversal that
#1077 identified as the dominant semilocal CUDA KS cost.

## Decision

For value-only exact direct execution, select J and K sources independently.
When a generated Coulomb owner is available and J uses ordinary FP64 arithmetic,
enqueue generated J first and then enqueue the existing independent kernel as a
K-only consumer on the same provider stream. K remains the shared exact
full-range provider; no DFT-specific exchange implementation is introduced.

Mixed-J requests deliberately keep the generic combined J/K path because
substituting generated FP64 J would change the requested arithmetic. Plans
without generated capacity or generated angular coverage also retain the
generic J/K fallback. K-only requests remain generic K-only.

This decision changes only the resident device-pointer value path used by CUDA
KS/Fock execution. Staged host-vector execution and analytic derivative
contractions keep their established exact paths.

## Rejected alternatives

- Do not recover J by subtracting exchange from the HF combined Fock result.
  J and K remain typed independent outputs.
- Do not introduce a hybrid-specific J/K implementation. The common provider
  remains authoritative for PBE0/B3LYP and other full-range consumers.
- Do not route mixed-J through generated FP64 J; its precision contract is
  intentionally distinct.
- Do not remove the generic J/K path. It remains the bounded fallback for
  unsupported generated coverage and capacity.

## Invariants

- Generated J and generic J implement the same exact full-range Coulomb
  operator and screening tolerance.
- Exact K keeps the existing FP64 independent exchange contraction.
- RKS/UKS spin semantics, nonsymmetric public density compatibility, and
  selected-output buffer ownership are unchanged.
- Generated J and generic K are ordered on the same provider stream; no new
  host transfer or synchronization is introduced.
- Derivative execution is unchanged.

## Evidence

The CUDA provider test pins the dispatch policy for generated hybrid, mixed-J,
bounded fallback, pure-J and K-only requests. Its independent full-ERI matrix
checks now exercise generated-capable J+K and generic-fallback J+K for both spin
modes and Cartesian/spherical coverage already present in the test matrix.

The change removes generic Coulomb work from a generated-capable hybrid value
build; it does not claim an endpoint speedup until allocated-GPU CI or benchmark
evidence is available.

## Consequences

Global-hybrid CUDA SCF can reuse the generated Coulomb work introduced for
semilocal KS while paying the generic quartic traversal only for exact exchange.
The extra source composition adds one ordered kernel sequence but no additional
persistent allocation beyond the already prepared optional Coulomb owner.

The next exchange-specific optimization can specialize or replace the remaining
generic K-only contraction without reopening Coulomb ownership.

## Revisit when

Revisit this split if generated exact-K shell consumers become available, mixed-J
gains an equivalent generated arithmetic contract, or profiling shows the
ordered generated-J plus generic-K sequence is slower than the combined generic
path for a well-defined small-work regime.

## References

- #1187
- #1077
- #1086
