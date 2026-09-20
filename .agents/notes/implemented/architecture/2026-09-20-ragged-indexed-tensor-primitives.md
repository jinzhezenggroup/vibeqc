# Decision: explicit ragged and cross-space TensorIR primitives

Status: implemented
Date: 2026-09-20

## Problem

An orbital-to-shell or pair-to-atom mapping changes a semantic index space.
Ordinary gather preserves its source space and cannot describe that change
honestly. A dense incidence matrix also hides sparse topology, adds storage and
makes reverse-mode accumulation needlessly dependent on dense contraction.

## Decision

Use `indexed_gather`, `scatter_add` and `segment_sum` as shared TensorIR nodes.
The mapped axis has an explicit output `Index`; all other axes retain their
source semantics. Positions are local to the mapped source or destination axis,
not absolute coordinates in its `IndexSpace`. Topology is an immutable tuple,
serialized and hashed with the mathematical program. This slice does not accept
runtime-varying topology or differentiate integer maps.

`indexed_gather` permits repeated source positions. `scatter_add` accumulates
repeated destination positions instead of overwriting them. `segment_sum` uses
monotone offsets spanning the source axis; repeated offsets represent empty
segments and contribute zero. Invalid extents, positions and offsets fail closed.
No ordinary gather may silently masquerade as a cross-space mapping.

## Derivative rules

For fixed topology, all three operations are linear in their floating input.
Their JVP applies the same primitive to the input tangent. In reverse mode:

- gather and indexed gather accumulate the output cotangent with `scatter_add`;
- scatter-add gathers the destination cotangent at each source position;
- segment-sum gathers each segment cotangent back to all its member positions.

Repeated indices therefore accumulate in the adjoint; they are not deduplicated
or assigned a last-writer result. Reference NumPy AD and generated derivative
DAGs implement the same contract. Removing the old dense gather-transpose matrix
is deliberate and must not change the Euclidean adjoint convention.

## Storage, lowering and execution boundaries

The CUDA planner owns allocation/alignment and lifetime accounting for emitted
static integer topology tables. The integer payload is included in the arena;
it is not unaccounted caller storage. The compiler owns map validation, formulas
and emitted reductions; native runtime owns allocation and execution. No SCF or
method policy belongs in these generic nodes.

The current CUDA lowering assigns each output a deterministic source traversal
rather than using nondeterministic floating atomic updates. Scatter-add can
scan the mapped input axis for every output position. Consequently bounded
storage does not imply linear total work or a production performance win.
A future inverted-index or parallel reduction schedule needs independent
numerical, resource and complete-consumer qualification before promotion.

## Rejected alternatives

Same-space gather would discard the target semantic domain. A dense incidence
matrix would turn sparse static topology into quadratic storage/work. Silent
runtime topology mutation would invalidate program identities. Atomic scatter
without an explicit reproducibility policy would change reduction semantics.

## Evidence and limits

The PR tests cover invalid topology, serialization, empty segments, derivative
duality, planned storage and source generation. An additional review sweep runs
72 combinations of rank 1/2/3, every axis, all three operations, FP32/FP64 and
empty/nonempty input. It compares explicit-loop references with interpreter,
reference/generated JVP and VJP, adjoint identities and serialization, and emits
CUDA plans. Repeated positions, nonzero mathematical index origins and Fortran
input layouts are included. These 72 host/source-generation checks pass on the
reviewed source; they are not 72 GPU executions.

The opt-in real-CUDA test and subsequent full-device consumer qualification
remain distinct from host tests and compilation. No public xTB endpoint,
all-device workflow or complete-endpoint speedup follows from this IR addition.

## Revisit when

Runtime-varying maps, higher-degree ragged consumers, larger scatter work counts
or a different deterministic-reduction schedule become necessary. Preserve the
same semantic/adjoint contract and include topology bytes in resource accounting.

## References

- PR #619; issue #501.
- `python/vibeqc_compiler/tensor/ir.py`, `types.py`, `autodiff.py`, `ad_program.py`.
- `python/vibeqc_compiler/tensor/cuda_plan.py`, `cuda_emit.py`.
- `tests/python/test_tensor_ragged.py`.

Agent: ChatGPT
Model: GPT-6 Astra Pro
