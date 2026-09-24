# Decision: qualify bounded Direct-HF as a complete endpoint

Status: implemented
Date: 2026-09-23

## Problem

Issue #945 mixed a structural-only descriptor diagnostic with numerical SCF
execution, then identified genuine failures after removing that diagnostic.
Restoring the generic Fock entry from #1055 and the diagnostic guard from #1069
was necessary but insufficient on current master. A further order-three cutoff
dropped fsss values, paged low-order forces omitted work counters, and the
single-system adapter converted NOT_IMPLEMENTED into INTERNAL_ERROR.

## Decision

- Keep screening purpose separate from the scientific consumer. A force kernel
  may use Fock screening; it cannot serve as the Fock output kernel.
- Drain uncovered Fock classes with the existing bounded hierarchical queue,
  excluding classes already owned by generated/native consumers. The scalar
  Fock drain ends at order two, while the scalar force drain ends at order three.
- Count each surviving low-order force page task at its point of consumption.
  This path bypasses the ordinary exact compactor's work ledger.
- Return NOT_IMPLEMENTED after tile validation, preserving it through both RHF
  and UHF single-system adapters. Diagnostic output is not numerical evidence.

No recurrence, tolerance, descriptor budget, or production CPU dependency is
introduced. In particular, the fix does not materialize an unbounded task arena.

## Evidence

Release/sm_120 on a Slurm-allocated RTX 5090 with CUDA 12.9.86:

- Eighteen GPU regressions pass with skips treated as failures: the five
  corrected issue reproductions in forced and default modes, independent
  Cartesian s/d/f RHF/UHF endpoints, fixed/bounded work parity, existing spd
  replay, and structural-only RHF/UHF diagnostics in both modes.
- Each high-l endpoint accounts for 55 shell quartets, 100 tiles, and 14,706
  AO/primitive quartets, including nonzero fsss work. PySCF energy errors are
  below 5e-14 hartree and force errors below 2e-11 hartree/bohr.
- A water-tetramer/def2-TZVP spherical batch of two naturally exceeds the
  one-GiB fixed descriptor budget. One system has 4,528,320 Cartesian tiles;
  two would require 1,738,874,880 bytes of fixed task descriptors. The automatic
  route completes and its final-density work is exactly twice the single-system
  fixed route. RHF agrees with independent PySCF well inside 2e-8 energy and
  2e-7 force gates.
- The corresponding cation doublet also agrees between fixed and automatic
  UHF. The symmetric PySCF initial guess is internally unstable; stability
  rotation, Newton minimization, and ordinary SCF refinement recover the lower
  state. An exact improper fourfold molecular symmetry maps its equivalent
  localized state to the CUDA state (coordinate residual below 2e-15 bohr).
  Independent energy/force errors are below 6e-13 / 1.1e-9 after that discrete
  mapping. Retain both the original force labels and the symmetry operation;
  comparing different degenerate localized states without the mapping is not
  a numerical failure of bounded execution.
- Nsight confirms bounded page/stream consumers and records launch, compaction,
  register/shared-memory and allocation baselines. These intrusive observations
  establish the route and resource baseline, not a speedup claim.

The retained evidence bundle supplies exact binary/source identities, raw
endpoint values, work counts, commands, and the dirty-source reconstruction.

## Rejected alternatives and revisit conditions

Rejecting every registry gap would make ordinary supported s/p/d/f topologies
fail after SCF setup. Calling a force wrapper with Fock screening corrupts the
output contract. Expanding the fixed arena defeats the required memory bound.
Retain the exact fallback until generated coverage and complete endpoint gates
justify removing it; optimize its work separately from this correctness slice.

## References

- #945, #1055, #1069.
- `tests/python/test_bounded_direct_high_l_cuda.py`.
- `benchmarks/results/bounded-direct-945-20260923/`.
