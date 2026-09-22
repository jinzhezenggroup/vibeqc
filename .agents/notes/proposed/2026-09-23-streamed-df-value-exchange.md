# Decision: qualify streamed DF occupied projection separately from response reuse

Status: proposed; implementation under qualification
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
- J remains two passes; total predicted J+K source work is 3.5 tensors.

These are schedule counts, not endpoint acceleration claims. Automatic factor
reservation can slightly change actual capacity. All four projection/raw buffers
remain within the existing value allowance; no response budget is borrowed.

## Evidence and remaining gates

- Emitted C++ and Python scheduling pass an independent enumeration of every
  legal row width on small nondivisible shapes, triangular and full K.
- The exact 96-atom shape selects the counts above without GPU allocation.
- Compiler dependency audit passes (321 modules).
- Allocated numerical, seed fallback, cold/warm/changed-geometry endpoint and
  resource qualification are pending at initial submission. Full 96-atom
  energy-plus-force completion is not established.

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
