# Decision: keep ProgramIR scientific identity separate from logical SPMD lowering

Status: implemented
Date: 2026-09-21

## Problem

ProgramIR described one finite synchronous provider schedule, but had no compiler
contract for multiple logical devices, sharded boundary ownership or explicit
collectives. Adding physical GPU topology directly to ProgramIR would have mixed
scientific/provider identity with deployment details and broken the stable schema-v2
serialization contract.

## Decision

Keep ProgramIR schema v2 unchanged and add a separate immutable `SpmdPlan`.
The plan binds one ProgramIR identity to a logical `DeviceMesh`, complete
per-buffer replicated/sharded placement metadata, and ordered explicit collectives.

The v1 collective set is intentionally bounded to all-reduce, reduce-scatter and
all-gather. All-to-all is deferred until a real VibeQC consumer requires it.
Sharding is limited to one axis of a buffer with an existing `DenseLayout`; opaque
buffers may be replicated but are not guessed into shardable shapes.
A size-one mesh canonicalizes collectives away and therefore provides the fallback
for the same scientific ProgramIR instead of selecting a second equation graph.
Balanced contiguous partitioning assigns remainders to lower logical shard ranks.

Physical device ordinals, GPU product names, links and topology are not ProgramIR
or SpmdPlan scientific facts. Runtime bindings may map logical ranks to actual
devices only after backend capability checks.

Communication scratch is represented through the existing resource planner.
Collective source-plus-result traffic is recorded as
`ResourceCandidate.relative_cost` so communication is not a free operation in
profitability selection. The value is an architecture-neutral work count, not a
latency/bandwidth prediction. Scatter/gather transitions charge the larger of the
pre- and post-collective local representations as scratch.

## Rejected alternatives

- Bump ProgramIR and add sharding fields directly to every buffer/call: rejected
  because deployment choices would perturb scientific identity and every current
  serialized consumer before a production multi-GPU endpoint exists.
- Infer sharding for opaque buffers from byte size: rejected because bytes do not
  prove tensor shape, ownership or legal partition axes.
- Add NCCL/CUDA execution in the common compiler module: rejected because compiler
  contracts must remain importable without runtime/GPU initialization.
- Treat collectives as zero-cost annotations: rejected because the scheduler would
  otherwise be able to prefer communication-heavy plans without accounting evidence.
- Add all-to-all preemptively: rejected because #834 explicitly requires a concrete
  consumer before expanding the collective surface.

## Invariants

- ProgramIR mathematical/provider identity remains independent of logical mesh size.
- SPMD identity includes mesh, placement, collective order and the bound ProgramIR.
- Every ProgramIR boundary buffer has exactly one explicit placement in a plan.
- Sharded buffers require a validated dense layout and deterministic partition.
- A backend must declare every required collective before execution.
- Publication requires the exact complete logical shard set; partial completion
  cannot silently become a scientific result.
- No hardware product name or physical topology assumption enters scientific IR.

## Evidence

`tests/python/test_program_spmd.py` covers 1-device fallback, 4-way ragged
partitioning, multi-axis logical coordinates, strict serialization/replay,
all-reduce/reduce-scatter/all-gather CPU reference semantics, unsupported-backend
failure, communication resource/cost accounting, and partial-shard publication
failure. Existing serial ProgramIR tests remain unchanged.
Validation used:

```bash
PYTHONPATH=python:. python -m pytest -q \
  tests/python/test_program_spmd.py tests/python/test_program_ir.py
# 61 passed

cmake -S . -B build-834-cpu -G Ninja \
  -DVIBEQC_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-834-cpu -j10
VIBEQC_LIBRARY=$PWD/build-834-cpu/libvibeqc.so PYTHONPATH=python:. \
  python -m pytest -q tests/python/test_program_ir.py \
  tests/python/test_program_spmd.py tests/python/test_program_ir_xc.py \
  tests/python/test_xc_contractions_native.py
# 125 passed

PYTHONPATH=python:. python tools/check_compiler_structure.py
```

## Consequences

This slice supplies the reusable compiler contract but deliberately makes no claim
that an existing XC, SCF, integral or tensor endpoint executes on multiple GPUs.
A production consumer still needs a runtime collective binding, complete endpoint
resource accounting and measured scaling evidence.

## Revisit when

A bounded real consumer needs a placement feature that cannot be represented by
one dense tensor-axis shard, or measured backend evidence requires a richer cost
model. Add all-to-all only with such a consumer. Change ProgramIR itself only when
a scientific/provider semantic, rather than deployment metadata, truly requires it.

## References

- #834
- #682
- #203
- `docs/program_ir.md`

---

Agent: ChatGPT
Model: GPT-5.6 Sol
