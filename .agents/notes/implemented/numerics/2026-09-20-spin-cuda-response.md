# Decision: qualify spin CUDA response through the prepared Fock provider

Status: implemented
Date: 2026-09-20

## Problem

Issue #179 retained an explicit UHF CUDA/DF gate after the CPU spin operator
landed. The existing method-neutral Fock provider already accepts unrestricted
signed densities on CUDA, but its existence does not prove the response identity,
orbital action, multi-RHS or failure contracts.

## Decision

Add `CudaSpinJKBackend` around the existing unrestricted `FockPlan`. Exact and DF
share this owner. Retain the old RHF adapter gates rather than widening their
meaning. The UHF operator uses the optional spin J/K seam and preserves its
three-call generic CPU/oracle implementation. Its `validate_current` also runs
at the shared solver's zero-RHS/resource fast paths.

The backend exports a snapshot via the existing native SCF, not a second solver.
CPU overlap/hcore metadata and canonicalization are explicit preparation costs;
SCF and both verification Fock builds use the same CUDA source/metric. Snapshot
validation checks the physical commutator and canonical density/Fock drift.
Immutable reference identity and retained subspaces remain separate from the
borrowed source lifetime. A later explicit SCF solve does not mutate an existing
immutable snapshot.

DF mathematical identity binds the Fock plan and its retained rank. The plan
already binds the actual auxiliary shells/geometry and metric threshold. The
`prepared-spin-df:` namespace avoids asserting equivalence with a standalone
CPU `MetricFactor` hash that this owner has not consumed or verified.

## Rejected alternatives

- Removing RHF gates would advertise untested spin support in incompatible APIs.
- Creating new CUDA spin kernels would duplicate the existing unrestricted Fock
  provider and its source/resource policy.
- Returning total Fock minus hcore cancels tiny directions; use raw J/K instead.
- Relabeling a conventional or differently fitted snapshot violates stationarity
  and changes the response operator even when dimensions match.
- Calling this a resident solver would hide host transforms, arrays and Krylov.

## Invariants and evidence

- Alpha and beta exchange are separate; total-density Coulomb couples spins.
- Signed directions, including 1e-18 scale, are evaluated without SCF, hcore
  subtraction or CPU integral fallback.
- The explicit device budget bounds retained provider allocations, not total
  process peak. Host API payload bytes are not CUDA transfer measurements.
- The Slurm-only suite uses committed independent PySCF/Libcint direct/DF
  integrals, including f shells, and native H2+/LiH+/water+ references. It builds
  the explicit spin MO Hessian independently, checks cross-spin coupling and
  all shared multi-RHS strategies against true residuals below 1e-9.
- Failure tests cover invalid inputs, insufficient device/solver budgets, failed
  native SCF followed by successful replay, metric/occupation mismatch, and
  closure before zero-RHS solves.

### Qualification

- RTX 5090 / CUDA 12.9.1, Release sm_120 build with optional AOT shell kernels
  disabled: all 15 device tests passed in 19.01 seconds. Tests executed through
  Slurm `main`, `--gres=gpu:5090:1`, one node/task and a finite 14-minute limit.
  This is test-suite elapsed time, not a response speedup measurement.
- 42 existing UHF/problem/Krylov tests passed in 1.55 seconds; 19 native CPU
  RKS/UKS regression tests passed in 268.56 seconds. All applicable pre-commit
  hooks passed.
- The first real numerical run passed 14 tests. Its remaining failure was an
  overly narrow test exception expectation: the existing DF planner rejects an
  impossible budget with `ValueError`, while direct allocation uses
  `MemoryError`. Both reject the operation; the corrected test verifies valid
  native SCF/action replay after rejection. No numerical tolerance was changed.
- Initial environment-only runs never loaded the library: Conda Python's
  DT_RPATH and optional wheel runtime discovery selected an older cuSOLVER
  missing `cusolverDnXsyevBatched`. The successful run binds the CUDA 12.9.1
  nvJitLink/cudart/cuBLASLt/cuBLAS/cuSPARSE/cuSOLVER shared libraries before Python
  starts. This changes user-space library resolution, not Slurm device visibility
  or the scientific implementation. CUDA_VISIBLE_DEVICES remains untouched.

The J/K retained-byte diagnostic excludes optional native SCF/eigensolver caches
created during explicit reference export. Complete endpoint peak/transfer costs
remain a separate acceptance item; host payload counters cannot establish them.

## Consequences

This completes a host-orchestrated spin CUDA exact/DF action boundary, not native
CUDA CPKS, resident blocked Krylov, endpoint speedup evidence or issue closure.
Revisit the adapter when a native device-vector interface can consume both spins
without host staging; retain the same shared Krylov controller.
