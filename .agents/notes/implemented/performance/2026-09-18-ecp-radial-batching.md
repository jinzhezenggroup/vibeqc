# Decision: bounded ECP radial batches with ordered pair accumulation

Status: implemented
Date: 2026-09-18

## Problem and decision

The CUDA adapter launched AO evaluation, projection and pair contraction
separately for every radial layer. Small systems expose few projector threads.
The compiler now owns a four-layer schedule for at most 16 public AOs and
44 polar points; larger domains retain one layer. The same compiler policy
serves native header emission and Python resource estimates. Native runtime
owns allocation and launches; the final batch uses its actual layer count.

AO evaluation and projection visit every radial source once. Pair contraction
assigns one thread to each unique AO pair, visits radial layers in ascending
order, and retains every per-layer atomic addition of the generated FP64
value and A/B/C derivative expressions. Center mapping and reduction order
remain unchanged. Neither the quadrature grids nor tolerances change.

## Lifetime and bounded fallback

Values and projections share one owned allocation with disjoint ranges. Only
a CUDA OOM from optional batched staging permits a one-layer retry. A failed
allocation owns no storage; its CUDA error is cleared before retry. Host
metadata failures, other CUDA errors and minimum-layer OOM propagate. The
ledger retains rejected attempts even when the fallback succeeds. Planned
budgeted endpoints include four-layer capacity and need no rejected attempts.

The CPU workspace is unchanged. CUDA selects the schedule by public AO count
but bounds storage using Cartesian count, including the 16-public/19-Cartesian
spherical-f case. KS one-electron setup shares that bound without gaining any
new method capability. Staging remains independent of radial node count.

## Rejected alternatives and invariants

Parallel radial pair threads or tile-local pair sums would change FP64
addition order; neither is necessary. Unrestricted tile promotion would
multiply large-AO scratch without evidence. The original single-layer domain
and independent CPU oracle remain. No screening, interpolation, mixed
precision, source recomputation or CPU substitution is introduced.

## Evidence and revisit conditions

Tests cover a 161-node tail, the 44/45-polar boundary, larger f-shell fallback,
independent raw values/derivatives, complete HF forces and strict-budget
replay. Native tests force OOM with one-layer capacity and check non-OOM
errors, cleanup and recovery. See `benchmarks/results/ecp-schedule-171/`
for all-repeat complete endpoints, actual Nsight launch counts, identities
and sanitizer results. Extend the bounded domain only after comparable
larger-system and memory evidence. Refs #171 and `docs/performance_engineering.md`.

On the RTX 4090, three-repeat medians for the 9-AO RHF/UHF fixtures decrease
20.7–20.8% for warm replay and 31.3% for changed geometry. The 16-AO fixture
decreases 3.58% / 5.45%. The 29-AO fallback's repeated timings remain within
1.5%, but its single first-call sample increases 6.55%; retain that regression
and avoid universal performance claims. All corresponding samples preserve
iteration counts and pass independent energy/force gates (maximum errors
`9.326e-15` Eh / `1.095e-9` Eh/bohr). Actual raw launches decrease 483 to 123
without changing source work. Contraction itself slows while projection and
complete endpoints improve, so isolated contraction timing is not the gate.

For 9-AO RHF, measured owned-device peak grows from 1,616,872 to 4,970,056
bytes; for 16 AO it grows from 3,038,560 to 8,999,776 bytes. The budgeted
samples fit their conservative plans without rejection. An initial native
OOM test budget omitted the full sphere-harmonic storage and failed before
one layer could fit; using the actual sphere element size fixes the fixture.
Both final OOM/recovery and tail-batch sanitizer runs report zero errors.

Native code adds 18 physical lines (13 noncomment code lines); the adapter
remains conservatively scientific in the ownership ledger. Removing the
four-line compiler-emitted helper exactly recovers the baseline generated
header. The original ownership report retained other generated-family
measurements from the unchanged base build and used the candidate ECP emission.
There is no scientific-code retirement claim. Final source and binary hashes,
every endpoint sample, and the external raw archive digest are retained in
the evidence bundle, including the failed initial fixture log.

## Integration with f projectors

Master `1865c5a` adds the f-projector capability while this schedule was being
qualified. Integration retains the ordered radial loop and optional OOM
fallback while adopting master's compiler-owned `ecp_projector_count` for
all projection offsets and capacities. Resource bounds use 16 components,
and tail/grid-boundary tests now exercise both ordinary and f projectors.
Removing only the schedule helper from the merged generated header exactly
recovers the current master's header. The original timing and memory numbers
above stay bound to their original commits; merged-head integration evidence
is recorded separately in `benchmarks/results/ecp-schedule-171/merge-validation.md`.
The merged ownership report is regenerated from current sources and measures
the materialized ECP header; other unchanged generated-family entries are
retained from current master.

## Clarification: the production call site defines radial ordering

The review of PR #458 observed that the old internal kernel decoded a radial
index from the thread index. Taken alone, that body could permit competing
radial threads if launched for several layers. The actual production caller
in both `b78215e` and `1865c5a` instead passes `nr=1`, advances `dradii + r`
in an ascending host loop, and queues every launch on the same stream. It
therefore does not execute concurrent radial-layer atomics for an AO pair.

The new kernel serializes up to four layers inside the pair-owning thread;
its internal mapping changes, while the production sequence of per-layer
FP64 updates remains ascending. The same triangular pair ownership and
per-layer A/B/C updates handle coincident physical centers. This is not a
claim about hypothetical direct launches of the old kernel with `nr>1`, nor
a cross-architecture bitwise guarantee. See the source-linked
[ordering audit](../../../../benchmarks/results/ecp-schedule-171/reduction-order.md).
The existing independent numerical gates remain the scientific qualification.
