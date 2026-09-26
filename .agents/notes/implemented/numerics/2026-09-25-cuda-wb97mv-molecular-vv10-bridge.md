# Decision: admit molecular VV10 through the CUDA KS host AO bridge

Status: implemented
Date: 2026-09-25

## Problem

The internal CUDA KS host-unfused path can stage AO density and potential while
retaining a CUDA VV10 pair owner. Its molecular-domain admission still required a
CPU VV10 owner, so the complete WB97M-V comparison failed before pair execution.
That guard predated the resident molecular-domain CUDA path.

## Decision

Apply the existing MolecularV1 padding in the host AO integration bridge before
either CPU or CUDA VV10 pair execution. Keep the domain restricted to the VV10
variant. The caller's prepared pair owner performs the pair calculation; the CUDA
KS host path does not create or invoke a CPU reference owner. Public CUDA WB97M-V
continues to require the device-fused schedule, while the internal host-unfused
route supplies a bounded comparison endpoint.

## Invariants

Validate physical density, gradient, and weights before padding. Inactive points
have zero quadrature weight and finite dummy features. Preserve the prepared pair
extent and its work bound. Keep rVV10, forces, and unrelated nonlocal combinations
outside this admission.

## Evidence

The Slurm RTX 5090 `vibeqc_ks_cuda_tests` run compares complete RKS and UKS
WB97M-V energies and components from CPU reference, CUDA host-unfused, and CUDA
device-fused paths. It also exercises the public energy endpoint and its host
schedule and force gates. `vibeqc_dft_cuda_tests` qualifies the resident
semilocal E/V path. These are compact matched-grid cases, not broad chemical
accuracy or performance claims.

## Revisit when

The public host-unfused CUDA schedule is separately qualified, or a new VV10
density-domain policy changes the active-point semantics or pair-work bound.

## References

Supersedes the CPU-only molecular admission described in
`../architecture/2026-09-22-wb97mv-self-consistent-composition.md` for the internal CUDA host
bridge. PR #1332.
