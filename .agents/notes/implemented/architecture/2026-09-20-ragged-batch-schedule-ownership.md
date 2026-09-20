# Decision: one shared owner for ragged batch scheduling metadata

Status: implemented
Date: 2026-09-20

## Problem

Ragged TensorIR needs explicit work/degree metadata without duplicating its
static map layout in each backend. Concurrent scatter optimization moved CUDA
to inverted member lists and fixed a host allocation before budget admission;
a merge must retain those guarantees rather than restoring the earlier scan.

## Decision

BatchScheduleIR derives batch domains and per-step ragged work from immutable
TensorIR indices/maps. The shared batch_schedule module owns the exact
length and offsets/member payload. Existing internal CUDA helper names delegate
to this owner, so emitter and planner clients retain one implementation.
Planning and repeated resource queries use lengths only, not a materialized
list of target buckets. Actual payload construction belongs to admitted source
emission. Empty-source scatter keeps the already-qualified zero-result path
and zero index-payload allocation. Immutable precision metadata remains cached.

Each scatter destination visits its ascending source-member list. This preserves
the original per-destination addition order without introducing floating-point
atomics or another scientific formula. Segment sums retain contiguous traversal;
indexed gathers retain direct lookups. A changed map changes mathematical/plan
identity through the existing immutable TensorIR contract.

## Work and resources

The exact aligned offsets/member bytes remain in TensorPlan admission. Work
metadata distinguishes original scan visits from scheduled source visits and
records degree/empty-group distributions; it is not elapsed time, total process
memory, or a complete SCF/SCC scaling claim. Host code objects, driver state and
provider-private allocations retain their existing exclusions.

## Rejected alternatives and boundaries

Do not retain two independent CUDA and shared inverted-table builders. Do not
materialize a target-sized host map merely to decide that the plan is too large.
Do not remove the empty-source guard, lower-budget rejection, precision cache,
or deterministic ordering while resolving concurrent changes. No batching of
SCF convergence or method policy is introduced, and no production pair kernel
or default fusion schedule is selected by this metadata.

## Validation and revisit conditions

Admission regressions reject a huge sparse destination before payload construction
and verify metadata queries on feasible plans. Existing ragged, AD, precision and
identity tests remain the compatibility gates. GPU numerical/performance evidence
must retain its actual source and endpoint scope; host work counts alone do not
qualify a hardware speedup. Revisit scheduling only with independently validated
output/order semantics and complete endpoint evidence.

Agent: ChatGPT
Model: GPT-6 Astra Pro
