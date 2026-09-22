# Decision: native bounded CPU component enumeration

Status: implemented
Date: 2026-09-20

## Problem

Issue #171's CPU d-shell force path uses 362 generated kernels but enumerates
every ordered component and primitive in Python. The previous qualified NaH
case visits 196246 primitive records; warm complete calls still took roughly
12--16 seconds. Compilation remains a separate, much larger first-call cost.

## Decision

Keep the canonical derivative graphs, exact symmetry/axis binding and source
units. The compiler exports a finite immutable integer dispatch table. A small
native runtime enumerates products in the same order, applies the existing
normalization and ordered weights, invokes those same generated dispatchers,
and restores derivative slots. It reuses the fixed primitive tile, allocates no
heap storage and publishes a complete integral's output/count transactionally.
Python still owns ordered AO terms, TensorIR weights and atom scattering.

The original component executor remains explicitly selectable through the
private diagnostic's `component_execution="python"`. Errors never trigger a
silent fallback. Only bases containing d select the new consumer by default;
the existing s/p route is unchanged. Runtime libraries retain their callbacks'
owners. Source/header/toolchain/binary validation runs on every construction.

## Invariants

- Preserve component/primitive product order and FP64 arithmetic with contraction
  disabled. No screening, density-symmetry assumption or new integral recurrence.
- Execute every admitted primitive record, even for zero weights.
- Preserve exact physical-center and Cartesian-axis restoration.
- Account for the 10300 by 9 integer table, labels, pointers and construction
  copies inside the existing additional-host inventory.
- Keep existing work, source, dense-provider and staging admission limits.

## Rejected alternatives

Reimplementing derivative mathematics in the runtime creates a second scientific
owner. Materializing full derivative tensors removes bounded streaming. Skipping
zero weights changes semantic work and is unnecessary for this scheduling change.
CUDA higher-angular force promotion requires independent NVIDIA numerical
evidence and is separate from this CPU slice.

## Evidence

`test_first_derivative_schedule.py` exhaustively checks dispatch metadata.
`test_component_streaming.py` checks mid-tile native failure, unchanged output
and count, recovery, invalid metadata, immutable integer arrays and header cache
invalidation. The Cartesian/spherical d-force suites retain independent Libcint
weighted energy differences, complete analytic gradients and two reconverged
finite-difference steps. Candidate tiles 1/2/128 compare with the Python executor.

Paired PBE-RKS NaH+d / H2 ragged batches compare complete baseline/candidate
cold-batch, warm and changed-geometry calls under exact resource budgets, with
failure/recovery and actual primitive/quartet/ECP-work equality. Cold batch means
a new prepared SCF owner; generated code is already in the shared cache. CI
retains measurements in the existing `benchmark-debug` artifact. Performance
promotion is limited to this declared endpoint and requires measured improvement;
there is no claim of faster first-call compilation or broader ECP capability.

## Revisit when

Generated compilation dominates user workloads, AO enumeration becomes the next
bottleneck, or a larger/angular domain warrants a different bounded schedule.

## References

- Issue #171; prior CPU d-force PR #658.
- [Prior schedule](2026-09-20-spd-cpu-derivative-schedule.md).
- [Current contract](../../../../docs/ecp.md).
