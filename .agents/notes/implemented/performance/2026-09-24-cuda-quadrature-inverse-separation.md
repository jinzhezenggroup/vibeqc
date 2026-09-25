# Decision: reuse inverse center separations in CUDA Becke partitioning

Status: implemented
Date: 2026-09-24

## Problem

The CUDA molecular quadrature computes each atom-pair separation once during grid setup, but the point-heavy Becke partition subsequently divides by that same separation for every ordered `(point, a, b != a)` visit. For the retained standard-PBE endpoint shapes this means billions of repeated FP64 divisions even though the denominator depends only on the unordered atom pair.

At 48 atoms / 1,327,104 points the partition performs 2,993,946,624 ordered non-self pair visits. At 96 atoms / 2,654,208 points it performs 24,206,376,960. The center geometry contains only 1,128 and 4,560 unordered non-self pairs respectively.

## Decision

Keep the existing `atoms * atoms` geometry buffer and change its meaning from center separation to inverse center separation. `geometry_kernel` evaluates the physical separation once per unordered pair, applies the existing `coincident_tolerance` test at that same setup boundary, and stores either `1.0 / separation` or the exact zero sentinel. The point-tile partition kernel then multiplies the distance difference by the retained reciprocal.

The existing scalar Becke graph, pair orientation, clipping, log/log1p accumulation order, point/atom traversal, normalization, memory footprint, tile size, and host export are unchanged. No extra persistent or transient buffer is added.

## Rejected alternatives

- Keep dividing inside every point worker: numerically identical but retains the dominant redundant denominator work.
- Add a second reciprocal matrix while preserving separations: avoids changing the geometry-buffer meaning but doubles the `O(atoms^2)` geometry storage without any current consumer of the raw separation matrix.
- Precompute point/pair Becke factors: can also share work, but requires `O(tile * atoms^2)` storage or a substantially different reduction schedule and is not a bounded follow-up to the existing implementation.
- Use an approximate reciprocal intrinsic: rejected because it would introduce a much larger and target-specific numerical change. The implementation uses one ordinary FP64 division per unordered pair.

## Invariants

- The coincident/tolerance branch is decided from the original FP64 separation before taking its reciprocal.
- A stored zero means the historical `mu = 0` coincident/tolerance path.
- Pair orientation remains `hi - lo`; `log(pair)` and `log1p(-pair)` ownership must not reverse.
- Becke polynomial evaluation and per-atom accumulation order remain unchanged.
- CUDA grid points/weights must continue to pass the independent CPU `MolecularGrid` reference gates, including near-coincident centers and contracted partition-weight derivatives.
- Resource accounting remains byte-identical because the existing geometry matrix is reused in place.

## Evidence

Deterministic non-coincident FP64 division census:

| endpoint | partition divisions before | geometry divisions after |
| --- | ---: | ---: |
| 48 atoms / 1,327,104 points | 2,993,946,624 | 1,128 |
| 96 atoms / 2,654,208 points | 24,206,376,960 | 4,560 |

This is a work-count reduction, not a wall-time speedup claim. A source-matched CUDA profile must measure `partition_kernel` and complete molecular-grid preparation before promotion.

The numerical operation changes from one FP64 division per point/pair to multiplication by a once-rounded FP64 reciprocal. Therefore the independent CPU-vs-CUDA quadrature suite is a mandatory acceptance gate rather than assuming bit identity. `tests/native/test_cuda_quadrature.cpp` already compares CUDA points/weights against the scalar CPU triangular Becke reference, covers all partition iteration counts, partial tiles, 96 atoms, coincident and 5e-13 near-coincident centers, translation/permutation, derivatives, and exact resource accounting. The existing tolerances are not relaxed.

`tests/python/test_quadrature_schedule.py` additionally locks reciprocal construction, tolerance ownership, absence of the point-loop separation division, and the 48/96-atom work census.

## Consequences

The geometry buffer now carries inverse separations, so future generated quadrature consumers must not assume raw distances are retained there. Grid resource bytes and launch count are unchanged. The setup adds no divisions beyond one per unordered non-coincident center pair; those replace billions of point-loop divisions on large production shapes.

## Revisit when

Revisit if an independent numerical gate exposes unacceptable reciprocal-rounding sensitivity, if a future consumer genuinely requires the raw center separations after setup, or if profiling shows the partition division is not material on supported hardware. In the first case prefer a numerically justified compensated quotient strategy over an approximate reciprocal intrinsic.

## References

Refs #1100, #1102, #1117. Stacked after #1193 only to serialize edits to the same generated CUDA quadrature owner.

Agent: ChatGPT
Model: GPT-5.6 Sol

## Review correction: non-representable reciprocals

The [reciprocal-overflow correction](../numerics/2026-09-24-becke-reciprocal-overflow.md) refines the buffer encoding for legal zero/tiny coincident tolerances. Positive entries remain finite reciprocals, zero retains its sentinel meaning, and a negative entry stores the original separation when its reciprocal would overflow. Only that exceptional case uses the original pointwise quotient. The ordinary-geometry census above remains applicable; a nonzero entry must no longer be assumed to be a reciprocal without checking its sign.

Agent: ChatGPT (Odd-PR Review)
Model: GPT-6 Astra Pro
