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

The integrated repair passed all 17 matched Release/sm_120 configurations
against master `3a304eff`, including its landed #945 coverage repair. The measured
candidate is `2cf3dec3`; subsequent qualification changes update only evidence,
notes, and the intentional registry fingerprint fixture. Three ABBA cycles cover
all four endpoint phases with seven warm repeats for the RHF matrix, five for
the UHF OH holdout, and three for the larger water tetramer. The existing 2%
ceiling is unchanged: the worst phase ratio is 1.00987495. The original ragged
STO-3G batch-three resident changed-geometry ratio is 0.96973264. Every sample
passes its numerical gate; complete per-class work and iteration counts agree.

The separate final-source Nsight replay observes the generated psss Fock grid
drop from 1360 to 176 CTAs, matching the retired native worker's claim bound.
The generated worker uses 133 registers, 15112 bytes of static shared memory,
and zero local bytes per thread. Its five traced launches total 4.860174 ms,
versus 4.840173 ms for the native worker in the matched baseline. These intrusive
kernel times explain the launch policy; clean endpoint acceptance uses the
separate ABBA samples.

Twenty allocated-GPU regressions pass without skips, including independent
libcint RHF/UHF comparisons when psss/fsss generated Fock classes are disabled.
Those registry gaps must reach the generic order-one/order-three value drain,
while the scalar force adapter still owns orders zero through three. This is
the critical composition invariant with the #945 repair.

The metadata deliberately changes the registry source fingerprint. Freshly
regenerated sm_120/sm_90 legacy bundles confirm that all eight CUDA shards and
the registry header remain byte-identical; the fixture records the one changed
registry hash and its rationale. All 50 integral-contract tests pass.

Compact source/binary-bound evidence, every timing/error/iteration sample,
independent oracle outputs, exact work counts, and reproduction commands are in
`benchmarks/results/psss-fock-888-20260923/`. Raw traces and interrupted pre-merge
runs remain ignored local artifacts. This is a structural-retirement
non-regression gate, not a statistically significant speedup promotion.

Preserve actual task-claim units when extending this metadata to other
schedules. In particular, Rys force and Fock schedules can differ. Compiler
metadata tests cover a packed row whose explicit Fock override uses one task
per CTA. Use complete endpoint evidence before changing other schedules' grids.

## References

- #888, #356.
- `.agents/notes/proposed/2026-09-21-psss-fock-retirement.md` (original proposal).
- `tests/python/test_fock_launch_metadata.py`.
- `python/vibeqc_compiler/integral/production_registry.py`.
