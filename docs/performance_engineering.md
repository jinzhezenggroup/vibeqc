# Performance engineering

VibeQC performance work must optimize scientific work and data movement, not
only kernel throughput or peak scratch size. This document records cross-cutting
rules for CUDA, generated integrals, response, DFT, and post-HF execution.

## Work amplification is a first-class metric

A schedule can satisfy a memory budget while repeating expensive work many
more times than necessary. Planners and benchmarks should therefore report both
memory and work, including the relevant subset of:

- consumer tiles/panels and source tiles/passes;
- integral/source evaluations and generated values;
- GEMM/contraction counts and their dimensions;
- bytes generated, gathered, uploaded, downloaded, or regenerated;
- synchronization/wait counts; and
- complete cold, warm, changed-geometry, and batch wall time.

When system size or a smaller budget changes a tile shape, compare actual work
counts before interpreting a kernel slowdown. A sudden endpoint cliff often
means the executed algorithm changed its amount of work.

## Prefer source-driven reuse

Expensive source work should normally be produced once and consumed by multiple
outputs before eviction. In particular, inspect loops of the form

```text
for consumer_tile:
    for every source_index:
        expensive_source_or_projection(source_index)
        consume(consumer_tile, source_index)
```

If the expensive operation does not depend on `consumer_tile`, either move it
outside that loop, retain/batch it under the resource policy, or demonstrate
with endpoint evidence that recomputation is faster.

The same rule applies to AO integral tiles feeding many MO blocks, auxiliary
projections feeding many response panels, grid/AO jets feeding several XC
consumers, and intermediates feeding CC/response equations.

## Reuse resident state before recomputation

Prepared plans often already own large, charged buffers and immutable metadata.
A downstream phase may borrow them when all of the following are explicit:

- scientific identity and generation match;
- the producer no longer needs the contents being overwritten;
- stream/event ordering protects the lifetime;
- resource accounting records borrowed capacity without double counting; and
- an explicit bounded fallback remains available when borrowing is impossible.

A component-local scratch allowance is not a reason to recompute a much more
expensive quantity if compatible resident storage is already owned elsewhere.
Memory safety, ownership, and work efficiency must be planned together.

## Keep oracle work out of production paths

Independent CPU/reference implementations are valuable correctness oracles,
but production CUDA setup must not construct transformed tensors, factorizations,
or derivative data solely because the oracle uses them. Record which consumer
owns each preparation result, and skip unused compatibility work on production
backends.

Likewise, avoid GPU -> host -> GPU staging when a CUDA consumer can use the
producer device data directly. Host-staged providers may remain explicit oracle
or compatibility paths, but performance claims must identify which route ran.

## Derivative and response consumers

Prefer consumer-driven derivatives:

```text
generated derivative primitive
    -> apply current response/adjoint weight
    -> bounded reduction
    -> physical gradient/HVP output
```

Avoid full coordinate-major derivative tensors when a generated kernel can
contract the final weight directly. Reuse common geometry, Boys, moment, or
shell work across requested derivative components when doing so improves the
complete endpoint rather than only an isolated kernel.

## Case study: PR #373

At 768 orbital and auxiliary AOs, a 128 MiB DF-response allowance split the old
response into 77 auxiliary panels. The panel loop recomputed every
`R_Q = D^T A_Q D` for every panel and reread the same raw three-center data.
That produced 118,272 AO projection GEMMs and about 282.6 GB of raw H2D traffic.

PR #373 reused three already-owned resident J/K temporaries, uploaded raw A
once, computed each Q projection once, and contracted the complete auxiliary
response with BLAS. The same 128 MiB of new response scratch then required
1,536 projection GEMMs and about 3.62 GB raw H2D traffic. On the qualified RTX
5090 case, the complete warm energy+force endpoint changed from 114.804 s to
12.083 s while retaining the existing numerical gates and bounded fallback.

The lesson is not that every calculation should allocate a full resident tensor.
It is that a planner must account for repeated scientific work, and should reuse
already-owned resident capacity when that is the faster valid execution policy.

## Performance qualification checklist

Before promoting a new default or auto-selection policy:

1. Record the baseline endpoint and exact scientific settings.
2. Record actual work counters for baseline and candidate.
3. Explain any change in algorithmic work caused by tiling or memory limits.
4. Validate numerical equivalence with an independent oracle/reference.
5. Test at least one larger size that can expose a planner/tile cliff.
6. Report cold, warm, and changed-geometry behavior when geometry-dependent
   preparation exists; report batch and constrained-memory behavior when those
   modes are supported.
7. Keep slower/correct fallbacks available where the promoted resource or
   identity preconditions do not hold.
8. Prefer complete endpoint evidence over isolated kernel speedups.
