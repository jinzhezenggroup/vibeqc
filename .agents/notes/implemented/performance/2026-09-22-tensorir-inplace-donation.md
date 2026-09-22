# Decision: make TensorIR in-place donation explicit and fail closed

Status: implemented
Date: 2026-09-22

## Problem

The shared #831 storage analysis can reuse a slot only after the previous owner is
dead in an earlier phase. That correctly rejects aliasing by default, but it also
misses a narrower case: an elementwise consumer may safely overwrite one
compiler-owned input at the same operation where that input is read for the final
time. The TensorIR CUDA arena therefore retained both the dead input and new
output through that phase even when the generated kernel has exact per-element
read-before-write semantics.

Treating all same-phase lifetimes as ordinary non-overlap would be unsound.
Donation must be an explicit operation contract rather than an allocator guess.

## Decision

Add an explicit donation edge to the backend-neutral storage analysis. A donation
is legal only when:

- the operation declares explicit memory effects;
- donor and recipient are physical storage owners rather than views;
- both are compiler-owned reusable ranges in the same memory space;
- capacities are equal;
- the donor's final use and the recipient's production are the same operation;
- one donor transfers to at most one recipient and vice versa.

A declared donation removes only that donor/recipient interference edge, reuses
the donor slot deterministically, and counts the same physical bytes once at the
transfer phase. Unknown aliases, opaque effects, external ownership, later donor
uses, capacity mismatch, and malformed donation edges all fail closed.

TensorIR exposes the capability as the opt-in
TensorSchedule(inplace_donation=True) schedule dimension. The first production
consumer is materialized elementwise CUDA. The planner donates only a direct,
materialized, non-pinned operand whose final use is the current elementwise
operation and whose shape, dtype, and aligned capacity match the result.
Inputs/constants/named earlier outputs are never donors.

Producer-layout optimization is deliberately incompatible with this first slice.
layouts=True plus donation fails before compilation because layout equivalence
has not yet been qualified for aliased input/output storage. Virtual operands are
also excluded so a transpose/slice mapping cannot introduce cross-thread
read-after-write hazards.

## Rejected alternatives

- Infer donation from liveness alone. Same-phase overlap normally represents
  a real read/write hazard. Reuse without an operation contract would be unsound.
- Donate through virtual views. A view may map output index z to a different
  physical input index, so another CUDA thread could overwrite data still needed.
- Enable donation with producer-layout search immediately. Equal byte capacity
  does not prove identical physical index mapping. This remains fail closed until
  layout compatibility participates in the donation contract.
- Turn ProgramIR into a runtime allocator. The common layer records storage
  legality and deterministic slot assignment only; native allocation/execution
  ownership remains unchanged.

## Invariants

- Baseline schedules retain their existing non-aliasing execution behavior.
- Donation is opt-in and participates in schedule/artifact identity.
- Scientific equations, operation order, finite/domain checks, and public output
  semantics are unchanged.
- Only compiler-owned temporaries can be overwritten.
- Unknown alias/effect behavior disables donation rather than weakening safety.
- The CUDA kernel must consume exactly the same logical element it overwrites;
  the initial TensorIR consumer therefore excludes virtual operands and layout
  optimization.

## Evidence

CPU/compiler regression:

    PYTHONPATH=python:. python -m pytest -q       tests/python/test_storage_analysis.py       tests/python/test_program_storage.py       tests/python/test_program_ir.py       tests/python/test_tensor_cuda_plan.py       tests/python/test_tensor_cuda_search.py       tests/python/test_streaming_schedule_compatibility.py       tests/python/test_geometry_pair_ir.py       tests/python/test_cc_triples_response.py

    190 passed

The 64-atom GeometryIR -> PairIR -> TensorIR inverse-power production graph
reduces TensorIR arena allocation from 195,072 B to 146,688 B, a
48,384 B (24.8%) reduction, while preserving the same program.

RTX 5090 real-device execution used CUDA 13.0 / sm_120 and the opt-in schedule
with elements_per_thread=4. The generated aliased elementwise chain matched the
TensorIR interpreter on repeated profiled/unprofiled executions:

    1 passed in 4.21s

A real three-atom PairIR inverse-power energy + generated coordinate-VJP CUDA
qualification also passed against the TensorIR interpreter with
views+fusion+donation enabled. The VJP selected four legal donations; this small
case did not reduce its arena high-water mark, which is intentionally not reported
as a memory win.

A small RCCSD(T) response graph finds many legal donation transfers but does not
lower its arena high-water mark. That result is retained as evidence that donation
count is not itself a peak-memory claim.

## Consequences

The Tensor CUDA plan schema advances to v5 because schedule/artifact identity now
includes the donation choice and each materialized step records its selected donor.
Search keeps donation qualification-only (False by default); promotion requires
complete-endpoint evidence rather than the memory win alone.

The shared storage analysis can now represent same-operation ownership transfer,
which future ProgramIR/runtime consumers may use only after they expose equally
explicit mutation/alias contracts.

## Revisit when

- producer-layout equivalence is strong enough to prove donation-safe physical
  index mappings;
- another subsystem exposes an explicit in-place provider contract;
- complete endpoint measurements justify adding True to the default search
  space;
- donation across unequal capacities becomes useful enough to justify explicit
  subrange/alignment semantics.

## References

#831, #682, #844, #863, #900.

---
Agent: ChatGPT
Model: GPT-5.6 Sol
