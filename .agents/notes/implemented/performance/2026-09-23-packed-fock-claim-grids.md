# Decision: bound generated packed Fock grids by queue claims

Status: implemented
Date: 2026-09-23

## Problem

PR #888 retires the handwritten fixed psss Fock worker. Its original ragged
STO-3G batch-three changed-geometry endpoint reproducibly exceeded the existing
2% retirement ceiling despite numerical and semantic-work parity. Nsight
located the difference in the generated psss Fock consumer, rather than task
classification or materialization.

The native packed worker capped its grid at ceil(task capacity / 32). The
generated route instead treated the individual-task capacity as a CTA bound.
Each generated CTA also claims 32 tasks, so the extra blocks wait for residency
only to discover that the persistent queue is already empty. Small molecular
topologies amplify this overhead across SCF and changed-geometry iterations.

## Decision

Emit the packed Fock task-claim width as registry metadata and use it when
bounding the fixed Fock grid. Both single-profile and multiple-profile
registries derive the width from the selected value schedule, including explicit
Fock overrides. Other schedules retain a conservative width of one and their
existing grid policy. There is no psss-name special case in the host runtime.

This changes empty worker launches, not task enumeration, screening, recurrence,
scatter, output reduction, or fixed/resident/paged ownership. The persistent
workers still consume every queued task. No task-sized buffer or host count
download is introduced.

## Rejected alternatives

- A first-order vector specialization reduced geometry state but still regressed
  the resident changed endpoint by about 3.9% in a fresh three-cycle ABBA test.
- Constant-work orbit stabilizer tests independently passed all 512 ordered
  RHF/UHF dense-ERI scatter checks, but the endpoint still regressed by 3.81%.
  That scatter change is not retained: this performance slice needs only the
  launch-policy repair.
- Restoring the handwritten psss formula would defeat the retirement objective.

## Evidence and invariants

The isolated grid repair passed the original fixture without relaxing its 2%
ceiling: three ABBA cycles, seven warm repeats, fixed and resident schedules,
all four phases. The resident changed ratio was 1.00333 and the worst phase
ratio was 1.00533. Energy/force errors were below 5e-13 / 4e-14 and complete
per-class work ledgers were equal. Final-source acceptance and integration
evidence accompany the PR; preliminary experiments are not substitutes for it.

Preserve actual task-claim units when extending this metadata to other
schedules. In particular, Rys force and Fock schedules can differ. Compiler
metadata tests cover a packed row whose explicit Fock override uses one task
per CTA. Use complete endpoint evidence before changing other schedules' grids.

## References

- #888, #356.
- `tests/python/test_fock_launch_metadata.py`.
- `python/vibeqc_compiler/integral/production_registry.py`.
