# Decision: Reuse the owner exponential in CUDA Becke normalization

Status: implemented
Date: 2026-09-24

## Problem

The CUDA molecular-grid normalization kernel uses a stable log-sum-exp reduction. For every molecular point it evaluates `exp(logs[a] - maximum)` once for every atom to form the denominator, then evaluates the exact same exponential a second time for the point-owning atom to form the numerator. The second evaluation is redundant and lies on the production grid-preparation path.

For the retained standard-PBE endpoint shapes this means one avoidable FP64 exponential per molecular point: 1,327,104 evaluations at 48 atoms and 2,654,208 at 96 atoms.

## Decision

Compute the point owner before the denominator loop and retain the owner's `term` while traversing atoms. The denominator still receives every term in the same atom order, and the final weight uses the retained owner term instead of re-evaluating `exp`.

No launch, allocation, grid prescription, point ordering, Becke polynomial, maximum reduction, denominator accumulation order, tolerance, or invalid-normalization handling changes.

## Rejected alternatives

- Fusing the maximum and exponential-sum passes with an online log-sum-exp update was rejected because it changes reduction arithmetic/order and would require a new numerical qualification for a much broader transformation.
- Materializing per-atom exponentials in scratch was rejected because the owner term is needed only once and can be retained in a register with no extra global memory traffic or resource accounting.
- A separate owner-only kernel was rejected because it adds a launch and repeats the same exponential work rather than removing it.

## Invariants

- The maximum over atom logs is computed exactly as before.
- The denominator terms are evaluated and accumulated in the same atom order.
- The point owner remains `(begin + p) / per_atom`.
- The final normalized weight, finite checks, invalid flag, tile schedule, and host export contract remain unchanged.
- CPU/reference grid code remains an independent numerical oracle.

## Evidence

Deterministic FP64 exponential work per complete grid normalization changes as follows:

| endpoint | before | after | removed |
| --- | ---: | ---: | ---: |
| 48 atoms / 1,327,104 points | 65,028,096 | 63,700,992 | 1,327,104 |
| 96 atoms / 2,654,208 points | 257,458,176 | 254,803,968 | 2,654,208 |

`tests/python/test_quadrature_normalize_schedule.py` locks the generated source shape and these work counts. Existing CUDA molecular-grid tests remain the numerical gate for point/weight agreement, partial tiles, coincident centers, partition iterations, derivative export, and resource accounting. Complete PBE grid-preparation/endpoints remain the performance/scientific acceptance gate.

## Consequences

The optimization has no additional persistent/transient storage and no extra kernel launch. It adds one owner comparison per atom term while removing one FP64 `exp` evaluation per molecular point. Wall-time benefit must be measured on matched Release CUDA builds; the deterministic work census is not itself a latency claim.

## Revisit when

Revisit if normalization is redesigned around a validated alternative reduction, if grid ownership no longer maps directly from public atom/radial/angular ordering, or if profiling shows a different normalization schedule is needed for complete-endpoint performance.

## References

- #1100: compiler-owned CUDA molecular quadrature
- #1102: CUDA XC/grid performance workstream
- #1117: GPU performance optimization tracker
- #1200: stacked quadrature radial-reuse base used to serialize same-file work
- `docs/maintainer/performance_engineering.md`

Agent: ChatGPT
Model: GPT-5.6 Sol
