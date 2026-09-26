# Decision: one low-angular DF Rys derivative family

Status: implemented
Date: 2026-09-16

## Problem

#394/#404 previously exposed only the one-root `000` lowering. Selecting
polynomial for the other six classes did not test their missing Rys candidates.
The batch now needs all seven classes under the same mathematical contract.

## Decision

Substitute root-dependent means and covariances into the existing Gaussian
moment IR. Apply the orbital-center raising/lowering identity at generation
time, prune each Cartesian component's moment graph, and immediately contract
its six independent derivatives with the existing folded response weight.
The existing finish/scatter recovers the auxiliary center by translation and
combines mathematical centers belonging to the same atom.

The node convention is `u=t²` with measure `exp(-T*t²) dt` on `[0,1]`.
The derivative degree gives `(li+lj+lk+1)//2+1` roots: one for `000`, two
for `001/002/100/101/110/200`. Availability is explicitly bounded to these
full-range first derivatives; it says nothing about automatic promotion,
range separation, higher derivatives, or higher angular classes.

All six added classes use the same two-root evaluator. An offline 90-digit
incomplete-gamma calculation constructs degree-17 Chebyshev interpolants on
24 intervals of width two. At `T>=48`, the generalized Laguerre limit replaces
the finite-interval quadrature; the relative F3 tail at the boundary is below
8e-18. Clenshaw evaluation avoids FP64 moment inversion and stays regular at
zero. Dividing nodes and square-root scaling weights avoids forming extreme
powers of T. Generation/production require neither mpmath nor an external
scientific runtime; the table's standalone reproduction tool needs mpmath.

## Rejected alternatives

The existing shared Direct two-root table was tried first. Independent
75-digit checks at `T=3.162277660168379e-7` found relative weight errors up to
about 9.8e-13, above this DF evaluator's 5e-14 node/weight/moment gate.
That table remains unchanged for its existing consumers. This family uses one
new shared table, not six class-specific root implementations. Do not relax
the DF root gate to accommodate the old table.

A generated DAG evaluated twice is not an independent oracle. Qualification
uses incomplete-gamma moment reconstruction, libcint contracted Cartesian and
spherical shell derivatives, native CPU derivative holdouts, and high-precision
center differentiation of the closed SSS integral. The last oracle also covers
extreme separation where FP64 coefficient convolution overflows. Such failures
of the polynomial oracle do not establish a failure of the new Rys evaluation.

## Invariants

FP64, normalization, external response semantics, force signs, shell schedules,
packets, screening, and the production manifest stay unchanged. Availability
and numerical qualification are distinct from performance promotion. All 42
class/lowering/schedule candidates are emitted in one batch. #404 owns one
combined-winner endpoint qualification against a pinned post-#411 baseline.

Work counters record one root evaluation per primitive, its actual root count,
and visited nonconstant moment states before backend CSE. Inactive folded
components perform no recurrence work. These are source-work counts, not CUDA
instruction counts or a prediction of elapsed time. Sparse aggregate ledgers
can bound recurrence work but cannot reconstruct the identity of every active
component; dense ledgers check the exact total.

## Evidence and consequences

The retained qualification bundle records numerical results, all candidate
identities, resource and binary growth, sanitizer status, and reproduction.
Production selection remains deferred to #404. Per-component moment DAGs may
use more registers at higher component counts; their measured slower results
are retained rather than replaced with unbounded manual per-class tuning.

References: #394, #404, #206; `docs/df_tuning.md`.
