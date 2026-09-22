# Decision: qualify streamed DF occupied projection separately from response reuse

Status: implemented; complete 96-atom benchmark qualification remains open in #1078
Date: 2026-09-23

## Problem

Issue #1078 retains a 900-second GPU HF-DF failure for 96 water-cluster atoms,
768 spherical AOs and 3712 auxiliary functions. The original Q=580 plan
generates two full raw tensors for J and seven for K on each Fock build.
PR #1089 reduces raw tile generation latency but does not remove these passes.

The existing `build_streamed_projected_exchange` can contract the occupied
index before metric whitening using four already charged buffers. Automatic
SCF currently excludes streamed plans at planning, allocation, selection and
seed qualification. Its shared resident predicate also authorizes final-state
and force-response borrowing, so relaxing that predicate would be incorrect.

## Decision

Add a compiler-owned shape/work schedule under `method/`. Native allocation,
value qualification and execution all use its emitted policy. Balance AO-row
blocks: in triangular K, early rows are regenerated for later blocks, so making
the first block as large as possible wastes work. Select the smallest row width
for the minimum feasible block count. The exact source census must beat the
existing dense fallback's census.

Authorize automatic factors for method-qualified singleton RHF streamed sources
only when the resulting four-buffer layout can use this schedule. Never turn an
otherwise resident dense plan into streaming or increase fallback source passes
just to reserve optional factors. Native execution additionally requires a
full-rank metric and allocated buffers. The existing exact spectral, discarded
Frobenius and full-density reconstruction tests remain unchanged; mixed or
unrepresentable seeds retain dense K.

Keep `qualified_resident_rhf_exchange` intact for final-state and force-response
leases. The new value-only qualification grants no such lease. Streamed factors
are private metric-eigendirection projections; they are not symmetric-C factors
and must not be exported or borrowed as if resident.

## Work and memory

For the original capacity `768*768*580` doubles per buffer and rank 160:

- Dense K requests seven raw tensors, 15,325,986,816 values.
- The prior explicit occupied path chose 576 rows and regenerated 1344 rows.
- The compiler balances two blocks of 384 rows and regenerates 1152 rows:
  3,284,140,032 values, equivalent to 1.5 full raw tensors.
- J remains two passes; an eligible SCF J+K build therefore predicts 3.5 tensors.
  The final physical K adapter still uses its existing dense fallback. Do not
  multiply this per-build prediction by iterations and call it endpoint work.

These are schedule counts, not endpoint acceleration claims. Automatic factor
reservation can slightly change actual capacity. All four projection/raw buffers
remain within the existing value allowance; no response budget is borrowed.

## Evidence and remaining gates

- Emitted C++ and Python scheduling pass an independent enumeration of every
  legal row width on small nondivisible shapes, triangular and full K.
- The exact 96-atom shape selects the counts above without GPU allocation.
- Compiler dependency audit passes (321 modules).
- Allocated GPU tests: 44 Python cases passed (43 in Slurm 11327, the corrected
  force/geometry test in a separate finite allocation), plus the native eight
  density-seed fixtures and final-state identity gates. The new source schedule
  is checked against independent libcint/PySCF J/K, including balanced/ragged
  panels and full/triangular traversal. Resource queries cover the practical
  768/3712 shape at four budgets and preserve resident/dense fallback behavior.
- At 12 atoms with forced streaming, cold/warm/changed-geometry energy-only and
  energy-plus-force solves pass absolute independent PySCF gates. Maximum
  observed energy and force differences are 4.67e-12 Eh and 1.12e-11 Eh/Bohr.
  Traces confirm exact rank-20 seed acceptance. Existing streamed response can
  consume separately validated canonical C with its **own** charged projection;
  it does not borrow the private value projection. An initial test incorrectly
  rejected this valid response route; the corrected test checks the actual
  final-projection lease and owned projection capacity, then passes all phases.
- Clean GPU execution comparison at 24 atoms / 192 AOs / 928 cc-pVDZ-JKFIT
  auxiliaries / 256 MiB value allowance: both controls stream with Q=79 and the
  same cold (15) and warm (2) iteration branches. Automatic cold execution is
  21.4769 s versus dense 51.2942 s; two warm samples are 4.6283/4.6317 s versus
  8.6153/8.6123 s. The comparator's independent GPU4PySCF energy gates cover all
  warm pairs: maxima 1.137e-12 and 4.547e-13 Eh. It does not serialize native
  cold energies, so these are not all-cold numerical claims. Neither timing is
  iteration-matched against the reference's one-iteration warm branch.
- Separate traced replay, Slurm 11337, confirms rank-40 seed acceptance and
  68,419,584 source values per occupied K (384 generated AO rows, three blocks,
  two full tensors), versus 410,517,504 values in the retained dense final K
  (12 tensors). Each J produces 547,356,672 raw bytes (two tensors). The trace
  distinguishes ordinary-stream operations from graph-capture descriptions;
  capture counters are not a count of replayed GPU work. Tracing is disabled in
  the clean timings above.
- Full 96-atom energy-plus-force completion remains unestablished. This repair
  reduces a source-work cause but does not close the original 900-second issue.

The real-GPU evidence is from the preserved combined integration based on
`60592ea9`, including #1086/#1089/#1091/#1096 and this change. It is not an exact
standalone PR-head measurement. Library SHA256:
`076cb5502e86983bb23d12bdf191542deb36b59be860fdddb631e18635e0baec`.
Local reconstructable source archive `integration-source-v7.tar.gz` SHA256:
`f90f56441201e4b4a22a2004ce92ec2a215ca39d8db4a40f68fca020f7873459`.
Artifacts under `.artifacts/gpu-blocker-fixes/` include `streamed-v7-gates.log`,
`streamed-force-v7-gates.log`, the corresponding pytest directories,
`hfdf24-streamed-{auto,dense}-v7.json` and progress journals, and
`hfdf24-work-v7.{jsonl,log}`. The archive precedes the final test-only correction;
the committed regression contains that correction.

## Rejected alternatives

- Relaxing the resident predicate globally would conflate value execution with
  final-state/response lifetime authorization.
- Tightening accuracy tolerances, changing the auxiliary basis or dropping
  metric directions cannot address a source-work cliff.
- Retaining raw plus whitened packed tensors does not fit the observed value
  budget; borrowing response capacity needs a separate audited live-set design.

## Revisit when

The full-rank restriction can be removed only with independent rank-deficient
metric qualification. Batch/UHF automatic selection and resident projection
reuse need separate resource/provenance evidence. Further J pass reduction or
retention of all projected rows may remove more source work after this route is
qualified; measure complete endpoints before promoting additional policies.

## Independent review integration

The current-master integration with the qualified KS/r2SCAN fixes passed all
four native DF suites (density seed, final snapshot, density-fitting integrals
and response, occupied response), including the 48 weighted finite-difference
gates. All 44 Python selector, streamed J/K, cold/warm/changed-geometry endpoint
and resource cases passed on RTX 5090. Source identity:
`3b8b92318f4f07be5aeb39c6753114be463640cde70b1950588950cda037be93`;
library SHA256:
`926a19f26198e36813deb6618367fa48eb0ca717e63d4f232fb2f009e2365cc7`.
This integration does not include the separate raw-mapping change in #1089.

The broad native suite also exposed stale response-work expectations, reproduced
with the preserved pre-change library. Resident RHF now uploads raw A once via
borrowed J/K scratch; resident UHF and generated dense/packed paths read their
whitened owner without raw traffic. A bounded streamed fitted response reads a
full raw tensor for each ordered auxiliary-panel pair; its `value_slices` counts
AO-pair source tiles, not auxiliary slices. The regression now protects these
exact work counts and keeps all independent force, capacity, transfer and
output-on-failure gates. Nonfinite host-raw rejection is checked on the RHF
route that actually consumes that input. No response production change was
needed, and full 96-atom completion remains unqualified.
