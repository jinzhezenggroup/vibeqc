# Decision: Keep exact UHF multi-RHS response vectors resident

Status: implemented
Date: 2026-09-21

## Problem

The shared UHF response operator already supported direct CUDA J/K actions and
the common sequential, blocked and recycled GMRES controller, but every Krylov
vector was host-owned. Extending the existing RHF resident owner by relabeling
it would be incorrect: unrestricted response needs two independent canonical
spin blocks and a coupled `J[Da+Db]`, `K[Da]`, `K[Db]` action.

## Decision

Add a tools-only `vibeqc_uhf_response_resident` C ABI and a matching
`CudaResidentUHFResponse` Python adapter. The owner borrows an exact,
unscreened unrestricted CUDA Fock plan, packs alpha rotations before beta
rotations in one device slot arena, and performs AO transforms, spin-coupled
J/K and orbital-gap assembly on the provider stream. The existing shared GMRES
implementation remains the only solver and is reused unchanged.

Density-fitted plans and KS/CPKS resident actions stay fail-closed. The owner
also rejects non-canonical spin rotation layouts so the native coefficient and
energy packing cannot silently disagree with the host problem identity.

## Invariants

- The parent Fock plan and source outlive the resident owner.
- Alpha and beta dimensions and occupations are validated against the immutable
  `UHFResponseProblem`.
- Successful solve iterations transfer only scalar dot/norm/status values; the
  final solution download is explicit.
- Empty beta occupied spaces are represented by a zero-dimensional beta block;
  its AO density is explicitly zeroed before the unrestricted provider call.
- Device allocation, logical solver workspace and retained recycle leases are
  separate bounded resources.

## Evidence

The CPU build compiles the additive ABI and the Python lifecycle tests cover
owner teardown, stale/released leases, parent closure and solver temporary
cleanup for both RHF and UHF adapters. The real-device gate is
`tests/python/test_response_spin_cuda.py` under Slurm; its exact UHF case
forbids host operator/J-K fallback, runs blocked and recycled solves against
an independent AO-integral matrix, and checks native upload/download/action
counters.

## Consequences

Exact UHF can now use the same resident multi-RHS resource model as RHF while
preserving spin coupling. A separate resident DF or semilocal XC action still
requires its own device ABI and independent numerical qualification.

## References

- Issue #179
- `docs/response.md`
- `tests/python/test_response_spin_cuda.py`
