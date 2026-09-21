# Decision: execute bounded CUDA solver regions through one shared runtime owner

Status: implemented
Date: 2026-09-21

## Problem

The generic `SolverRegion` contract from #835 can describe bounded loop state,
but the qualified CUDA KS experiment from #623 still owned its two-step
submission loop locally. Leaving that control owner inside DFT would create a
second lifecycle beside the shared CUDA replay infrastructure from #507/#520.

## Decision

Add `runtime::SolverRegionCudaExecutor` as a method-neutral native execution
owner. It bounds steps per checkpoint, records generic submission/checkpoint
metrics, and delegates optional replay to the existing `CudaGraphRegion`.
Scientific step semantics remain callbacks supplied by the method owner.

The direct all-electron RKS chunk from #370 is the first native consumer. Its
existing eligibility gate, convergence predicates, DIIS/history updates,
max-iteration handling, failure state, warm publication and scalar diagnostics
are unchanged. CUDA KS binds replay as disabled in this slice, so execution is
the already-qualified ordinary-stream bounded chunk rather than a new graph
performance path.

The shared binding exposes scalar and per-item-mask completion modes for future
consumers, but this change does not claim ragged-mask execution for KS.

## Rejected alternatives

- Keep the two-step loop in `cuda_ks.cpp`: that would preserve a DFT-specific
  control-flow owner after a generic region abstraction exists.
- Move convergence or DIIS into the runtime: those are method-owned scientific
  semantics and are not part of the generic executor.
- Enable CUDA Graph replay immediately: #370 requires complete endpoint and
  ragged-failure evidence before replay promotion.

## Evidence

Host-only lifecycle validation covers bounded submission, remaining-step
clamping, checkpoint accounting, fail-closed invalid requests, capture warmup,
capture, replay and invalidation through the shared graph owner.

The existing CUDA KS schedule regression verifies that the opt-in chunk policy
still carries the original physical gates. Native CUDA compilation and allocated
GPU endpoint validation are required before the change is proposed for merge.

## Consequences

CUDA KS no longer owns the generic bounded-loop submission stack. The repository
has one reusable native SolverRegion execution seam and one CUDA graph lifecycle.
Issue #370 remains open for replay activation, ragged completion qualification
and complete cold/warm/changed-geometry performance evidence.

## References

- #370
- #835
- #507
- #520
- #623
