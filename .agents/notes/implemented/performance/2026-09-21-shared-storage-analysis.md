# Shared whole-region storage analysis (#831)

## Decision

Add one backend-neutral `common.storage` analysis for compiler-visible ownership,
alias/view groups, live ranges, interference, peak live bytes and deterministic
scratch-slot assignment. ProgramIR and TensorIR adapt their existing facts into
this model; the analysis does not allocate memory, launch work, or own streams.

The first slice deliberately keeps execution unchanged. ProgramIR uses the shared
analysis as the source of its existing last-use resource intervals. TensorIR
exports the same analysis alongside its already-qualified CUDA arena planner so
the two models can be cross-checked before any allocator migration.

## Contracts

- `BufferValue` separates external/borrowed storage from compiler-owned storage.
- `AliasKind.VIEW` groups a logical view with one declared physical owner.
- `AliasKind.UNKNOWN` blocks early release and reuse for the whole memory space.
- `MemoryEffect.OPAQUE` pins every touched owner through the region boundary.
- Explicit operations remain ordered SSA-like producers; duplicate/forward writes
  fail before storage planning.
- Reuse is legal only across non-overlapping inclusive live ranges in one space.
- Slot capacity is at least the largest assigned owner and alignment is the
  maximum requirement of all assigned owners.
- Peak-byte accounting is based on simultaneously live ownership groups, not the
  sum of local subsystem peaks.

This contract is intentionally stricter than an allocator heuristic. Unknown
external alias/effect behavior cannot silently become an optimization.

## Consumers

### ProgramIR

`ProgramIR.storage_analysis()` adapts existing synchronous, disjoint boundary
buffers and explicit provider read/write declarations. Borrowed inputs remain
external and retained for the full region. `ProgramIR.lifetimes()` now consumes
the shared ranges, preserving the Phase-A release/resource semantics.

### TensorIR CUDA

`TensorPlan.storage_analysis()` adapts materialized CUDA arena values and qualified
zero-allocation transpose/reshape/slice views. Virtual broadcast/fused arithmetic
is flattened to its materialized dependencies rather than mislabeled as a dense
storage alias. The current arena offsets remain the runtime plan of record in this
slice.

## Evidence
Targeted CPU validation:

```text
PYTHONPATH=python:. python -m pytest -q \\
  tests/python/test_storage_analysis.py \\
  tests/python/test_program_ir.py \\
  tests/python/test_tensor_cuda_plan.py

62 passed
```

The ProgramIR fixture independently reports the same 218-byte simultaneous host
peak as the existing resource planner and assigns the dead `a` scratch slot to the
later `out` owner. TensorIR tests require shared-analysis peak live bytes to fit
inside the existing arena and require at least one reusable slot. A direct
transpose view is grouped with its materialized owner and receives no allocation.

## Not claimed yet

This slice does **not** claim #831 complete. It does not yet rewrite cross-subsystem
copies, propagate producer layouts across ProgramIR boundaries, enable in-place
donation, or replace the TensorIR/runtime allocator. The next production slice
should version an explicit ProgramIR alias/effect boundary and use it to remove a
qualified materialization beyond #460, with endpoint numerical/resource evidence.

Agent: ChatGPT
Model: GPT-5.6 Sol
