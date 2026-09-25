# Decision: reuse CUDA quadrature angular factors

Status: implemented
Date: 2026-09-24

## Problem

The generated CUDA molecular-grid point kernel evaluated the same angular special
functions for every atom and radial shell. For each molecular point it recomputed
one polar `sqrt`, one azimuth `cos`, one azimuth `sin`, and the azimuth weight
factor even though these depend only on the fixed polar/azimuth quadrature rule.

The qualified standard PBE production grid is 54 radial x 16 polar x 32 azimuth
points per atom. The retained 48- and 96-atom endpoints therefore materialize
1,327,104 and 2,654,208 points respectively, making this repeated point-level
special-function work visible inside CUDA grid preparation.

## Decision

Allocate two small generated-layout caches inside the existing transient CUDA
quadrature arena:

- three doubles for each of the maximum 256 polar entries: ring, node and weight;
- three doubles for each of the maximum 1024 azimuth entries: cosine, sine and
  angular weight factor.

Populate the active entries once on the same CUDA stream before point tiling and
make every point worker load those cached factors. The setup kernels use the same
device `sqrt`, `cos` and `sin` expressions as the old point kernel. Point ordering,
radial arithmetic, Becke partitioning, FP64 storage, host export and public grid
policy are unchanged.

The cache is fixed at 30,720 bytes so the pure shape-resource query remains
independent of a concrete `GridSpec` and no public/resource ABI grows another
shape parameter.

## Rejected alternatives

- Host-side angular precomputation was rejected because it would move the source
  of transcendental values from CUDA device math to the host libm and create an
  unnecessary CPU/GPU numerical-identity boundary.
- Recomputing the table once per 4096-point tile was rejected because angular
  factors are geometry-independent within the complete grid construction and can
  be retained for the whole operation at negligible memory cost.
- Constant-memory specialization by exact grid shape was rejected for this slice
  because it adds target/build specialization and another ownership path while
  the bounded arena already provides sufficient transient storage.

## Invariants

- Preserve atom-radial-polar-azimuth point order and existing FP64 arithmetic
  grouping in coordinate/weight assembly.
- Keep angular cache construction on the same CUDA stream before any point tile.
- Preserve the existing Becke equations, tolerances, partition orientation and
  normalization.
- Keep the resource query exact, including the fixed 30,720-byte cache charge.
- CUDA numerical qualification must continue to compare generated points/weights
  and downstream DFT results against the existing independent gates.

## Evidence

For the standard PBE topology (54 x 16 x 32), the old point kernel executes three
special functions per molecular point. The new setup executes 16 square roots and
32 cosine plus 32 sine evaluations once for the complete grid:

| endpoint | old sqrt+sin+cos evaluations | new setup evaluations |
| --- | ---: | ---: |
| 48 atoms / 1,327,104 points | 3,981,312 | 80 |
| 96 atoms / 2,654,208 points | 7,962,624 | 80 |

These are deterministic source/work counts, not wall-time speedup claims. A matched
NVIDIA run should compare complete molecular-grid preparation and `points_kernel`
time while requiring identical numerical acceptance. The implementation-time
worker was unavailable, so no unrun CUDA timing is represented as measured.

Focused regression coverage is in `tests/python/test_quadrature_schedule.py`; the
existing native CUDA quadrature and DFT endpoint suites remain the numerical gate.

## Consequences

Every CUDA grid construction pays two tiny setup launches and 30 KiB of additional
transient device storage. In exchange, point generation no longer scales angular
special-function evaluation count with atom or radial-point count. Cache loads are
regular: adjacent azimuth lanes read adjacent entries while a shared polar entry is
reused across each azimuth ring.

## Revisit when

Revisit this design if profiling shows the two setup launches dominate tiny grids,
if CUDA graph capture makes setup specialization profitable, or if the public grid
contract changes to pruning/screening that invalidates whole-grid angular reuse.

## References

- #1100 CUDA molecular quadrature
- #1102 CUDA DFT performance pipeline
- #1117 complete DFT endpoint qualification
- #1184 point-center 2-D CUDA schedule (stacked base for same-file ordering)
- `docs/maintainer/performance_engineering.md`

Agent: ChatGPT
Model: GPT-5.6 Sol
