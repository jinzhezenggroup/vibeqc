# Decision: keep direct streamed GEMM reduction opt-in

Status: implemented
Date: 2026-09-23

## Problem

The #845 streaming frontier bounds live high-rank TensorIR intermediates, but
an einsum that consumes a virtual producer can still lower to packed GEMM.
Packing then reevaluates that producer once per panel. On the fixed-capacity
RCCSD(T) runtime-domain graph, this amplifies launch and packing work.

## Decision

An explicit `TensorSchedule(stream_reductions=True,
streamed_gemm_reduction=True)` candidate routes einsums with virtual operands
through the existing generated reduction kernel. The shared planner makes this
choice before layout selection; the mathematical graph, CUDA emitter,
runtime-domain maps, error checks, and prepared artifact machinery are reused.
The new switch is keyword-only, changes the plan/artifact identity, requires
streaming, and is absent from the default search space. The default and #845
streaming schedules remain selectable.

This is a bounded memory alternative, not a persistent q-domain kernel. It
removes the repeated panel packing and the cuBLAS provider allocation for this
workload, but still reevaluates virtual producers inside generated reductions.
Do not promote it as the default or close #783 from this slice.

## Rejected alternatives

- Broad scalar inlining alone repeats more expensive producer work and was
  already slow in #845's RTX 5090 qualification.
- Materializing every shared producer restores substantial arena storage and
  does not directly address the packed panel launch count.
- A handwritten triples CUDA formula would create a second scientific
  implementation outside TensorIR ownership.

## Invariants

- The independent CPU tile path remains the numerical oracle.
- Runtime tile-map changes reuse one prepared artifact for a fixed capacity.
- Invalid index maps fail and a subsequent valid replay succeeds.
- The switch remains opt-in until a complete endpoint gate supports promotion.

## Evidence

On one NVIDIA H100 (80 GB), FP64, seed 783, full triangular domain, three warm
repeats per schedule, the matched `baseline / #845 streaming / candidate`
measurements were:

| `(nocc,nvir)` | Warm endpoint (ms) | CUDA device (ms) | Estimated launches | Planned peak bytes | Compile (s) |
| --- | --- | --- | --- | --- | --- |
| `(4,8)` baseline | 7.14 | 6.73 | 2,442 | 109,733,928 | 10.52 |
| `(4,8)` streaming | 204.84 | 203.51 | 12,970 | 105,117,992 | 27.31 |
| `(4,8)` candidate | 74.28 | 73.89 | 46 | 259,112 | 24.50 |
| `(8,16)` baseline | 43.58 | 42.80 | 14,970 | 363,770,696 | 10.45 |
| `(8,16)` streaming | 8,835.12 | 8,832.82 | 264,394 | 106,496,072 | 27.65 |
| `(8,16)` candidate | 1,177.99 | 1,177.39 | 46 | 1,636,168 | 23.67 |

All results agreed with the independent CPU tile oracle to relative error
below `3e-15`. Both shapes changed the runtime tile range on the same prepared
artifact, rejected an out-of-bounds map, then replayed the original range.
The H100 job completed with exit code 0. Planned peak is the TensorIR numeric
buffer bound, not an observed device high-water mark. Estimated launch count
comes from the emitted plan, not a hardware counter. The candidate reduces
the old streaming endpoint by about 2.8x and 7.5x, but remains about 10.4x
and 27.0x slower than the default endpoint. Its static flop estimate does not
count repeated virtual evaluation and must not be used as a work reduction
claim.

Raw results and the source archive/patch checksums are in
`benchmarks/results/tensor-streaming-783/generated-reduction-h100-*.json`.
Local gates: 190 passed, 1 CUDA skip for runtime-indexed/compiler tests;
12 passed, 1 CUDA skip for compatibility/lifetime tests; Ruff and the compiler
structure check passed. The CUDA skip is covered by the real H100 qualification.

## Consequences

The candidate offers a much smaller planned numeric buffer when the default
schedule cannot fit, at substantial endpoint and compilation cost. A future
cooperative or persistent q-domain lowering must share intermediate work
across output terms and qualify complete warm execution, not just launch count
or arena bytes.

## Revisit when

A generic q-domain kernel can retain or cooperatively produce shared W/Z
intermediates under a bounded budget, with independent AD and full endpoint
qualification on representative sizes and runtime tile changes.

## References

- Issue #783, PRs #790, #839, #845.
- `docs/maintainer/performance_engineering.md`.
