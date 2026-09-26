# Decision: invert static scatter maps and cache immutable TensorPlan precision facts

Status: implemented
Date: 2026-09-20

## Problem

The #504 GFN2 ragged geometry evidence exposed two unrelated compiler overheads.
On a 32 x 24 atom / 8,832-pair RTX 5090 workload, generated primal plus
repulsion VJP initially required about 2.44 ms of reported device time while
the pinned native CN + repulsion reference required about 0.366 ms.

The complete generated Python endpoints were much worse: about 522 ms combined.
This could not be attributed safely from the aggregate timing.

## Diagnosis

The existing deterministic CUDA `scatter_add` lowering assigned one output
thread and scanned the entire mapped source axis for every target. For pair to
atom accumulation this is O(atoms * pairs), exactly the scaling risk recorded
in the original #501 ragged-index note.

An isolated 8,832-pair / 768-atom CUDA comparison measured 0.84347 ms for the
scan lowering and 0.053184 ms for a static inverted-index lowering. The CUDA
outputs were bitwise identical because every destination retains ascending
source traversal.

The ~250 ms host gap was separate. Direct `tensor_run` wall time was 1.343 ms,
NumPy staging was 0.0012 ms and validation 0.0038 ms. Recomputing
`plan.semantic_traffic` cost 253.4 ms because it rebuilt the immutable precision
schedule on every execute; `plan.precision` added another ~31 ms outside the
reported endpoint timer.

## Decision

Keep the mathematical `scatter_add` IR unchanged. During CUDA planning, invert
its static destination map into a CSR-like table containing per-target offsets
and ascending source coordinates. Each output still owns its deterministic
serial reduction, so no floating atomics or new reproducibility policy are
introduced.

Cache `TensorPlan.precision` and `TensorPlan.precision_schedule` because a
TensorPlan is immutable. Execution metrics reuse those plan facts instead of
re-deriving scientific/compiler metadata on every numerical call.

Do not globally change the default TensorSchedule in this slice. Fusion is a
separate schedule decision, although #504 now provides strong evidence for a
fused GFN2 pairwise candidate.

## Evidence

Relevant host/source tests: 168 passed, 2 skipped. Allocated RTX 5090 ragged
TensorIR CUDA tests: 3 passed. Allocated GFN2 ragged primal/CN-VJP/repulsion-VJP
test: 1 passed.

After static scatter inversion and precision-fact caching, the same GFN2
workload measures:

- baseline primal: 0.16168 ms device / 0.23471 ms complete endpoint;
- baseline repulsion VJP: 0.18208 ms device / 0.26193 ms complete endpoint;
- fused primal: 0.094864 ms device / 0.16516 ms complete endpoint;
- fused repulsion VJP: 0.101712 ms device / 0.18026 ms complete endpoint.

Thus the baseline generated device total is 0.34376 ms and fused total is
0.196576 ms. The pinned xTBloom resident CN + repulsion reference remains
0.36552 ms. Endpoint boundaries are not identical, so this is replacement-gate
evidence rather than a universal cross-engine speedup claim.

## Remaining bottleneck

Runtime arithmetic is no longer the dominant problem. Generated static data is
still emitted as C++ literals. The fused #504 primal source is 2,212,634 bytes
and the fused VJP is 1,227,309 bytes; their NVCC compile times are 5.25 s and
4.99 s. The primal alone exceeds the default 2 MiB TensorIR tuner source budget.

GFN2 currently materializes ten pair-sized constant arrays in the primal,
including uniform constants and topology-repeated element parameters. Static
index maps are also rendered into source text. This should be addressed by the
compiler middle-end/static-data path rather than by restoring the slow scatter
algorithm.

## Consequences

The deterministic ragged primitive is now linear in its actual mapped source
work instead of output-size times source-size. Generated GFN2 pairwise kernels
are no longer rejected by runtime performance on this measured domain. Production
promotion still needs the runtime integration owned outside #504 and a decision
on static-data/source-size cost.

## References

- #501 and its ragged-index architecture note
- #504 / PR #697
- #168 for schedule/fusion policy
- #682 and #673 for broader compiler optimization
- `benchmarks/results/gfn2-geometry-504/performance-repair.json`

Agent: ChatGPT
Model: GPT-5.6 Sol
