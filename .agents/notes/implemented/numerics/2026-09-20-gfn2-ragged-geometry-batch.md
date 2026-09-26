# Decision: keep ragged GFN2 geometry in one TensorIR graph

Status: implemented
Date: 2026-09-20

## Problem

#504 needs CN and repulsion to prove that the compiler can own a real
geometry-dependent pairwise scientific slice, including derivatives and
heterogeneous molecules. A per-molecule Python loop would demonstrate scalar
reuse but would not prove ragged compiler ownership or one CUDA graph. The
generated path also needs a replacement decision against the existing
handwritten xTBloom kernels rather than assuming code generation is faster.

## Decision

Represent a heterogeneous molecular batch as one flattened atom population, one
canonical undirected pair population, and an explicit strictly increasing
system-atom partition. Pair enumeration is restricted to each partition, so
identical coordinates in different systems never create a physical pair.

CN scatters pair values to atoms. Repulsion scatters the same generated per-pair
expression to a batch index, producing one energy per system. Single-system and
ragged compilation share one `_gfn2_pair_terms` scientific owner. Cross-system
pairs fail closed, and any geometry change that changes a within-system 25-bohr
pair set invalidates topology identity.

Generated TensorIR is accepted as the scientific expression and derivative
source, but it is not promoted as the production replacement for xTBloom's
handwritten pairwise kernels at the current performance point.

## Rejected alternatives

A host loop over separately compiled molecule graphs was rejected because it
moves batch ownership out of the compiler and cannot prove one ragged CUDA
graph. Treating the flattened atom set as one molecule was rejected because it
would create cross-system pairs. Immediate native-kernel replacement was
rejected by RTX 5090 evidence: scientific gates pass, but the generated
device-side endpoint and source/compile costs materially regress the pinned
native path.

## Invariants

- Pair ownership is deterministic and never crosses a system partition.
- Scalar and ragged paths share one CN/repulsion scientific expression.
- The sharp 25-bohr topology participates in identity and stale-state checks.
- Coordinate derivatives come from TensorIR transposition.
- Performance promotion requires reproducible complete-endpoint evidence.

## Evidence

CPU qualification reports `11 passed, 2 skipped`; compiler structure reports
248 modules with zero dependency errors. An allocated RTX 5090 executes the
ragged primal and both generated coordinate VJPs and matches the interpreter
(`1 passed, 12 deselected`).

For batch 32 x 24 atoms with 8,832 retained pairs, generated primal plus
repulsion VJP produces 3,388,749 bytes of CUDA source, 4,949,560 bytes of
shared objects, and 12.0441 seconds of NVCC compilation. Unprofiled median
`device_ms` sums to 2.44858 ms/batch and complete host endpoints sum to
523.85249 ms/batch.

Pinned xTBloom revision `2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`
measures 0.18152 ms/batch for CN cache reuse and 0.184 ms/batch for repulsion
energy+force, or 0.36552 ms/batch resident term timing. The about-6.70x numeric
ratio is not a pure kernel ratio because endpoint boundaries differ; it is a
negative replacement gate, not a single-kernel slowdown claim.

## Consequences

The compiler now owns a real ragged pairwise scientific vertical slice and its
derivatives. Production native kernels remain in place, avoiding an unjustified
performance regression. Future work can optimize fusion, launch count, and host
staging without reopening the scientific ownership decision.

## Revisit when

Reconsider replacement when generated source/compile size is controlled and a
matched resident/native benchmark plus complete public endpoint benchmark shows
no material regression on representative ragged batches.

## References

- #504
- #621
- #655
- `benchmarks/results/gfn2-geometry-504/`

Agent: ChatGPT
Model: GPT-5.6 Sol
