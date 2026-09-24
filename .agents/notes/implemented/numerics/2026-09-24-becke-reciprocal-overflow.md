# Decision: preserve finite Becke quotients when a reciprocal overflows

Status: implemented
Date: 2026-09-24

## Problem

The optional quadrature optimization in #1197 changes pointwise division into multiplication by a retained reciprocal. GridSpec permits a zero coincident tolerance. A finite separation of 1e-310 then has an infinite binary64 reciprocal even though its point-distance quotient can be finite. At an equidistant point, zero times infinity becomes NaN and fmax/fmin clip it to -1 instead of the historical coordinate zero. The default 1e-12 coincident tolerance masks this particular boundary; no ordinary-molecule failure is inferred.

## Decision

Keep the existing geometry buffer with an explicit signed encoding: positive entries are finite reciprocals, zero retains the coincident/tolerance policy, and negative entries carry the original positive separation when the reciprocal is not representable. Only the negative branch uses the historical pointwise division. This preserves the normal reciprocal fast path and needs no additional storage or launch.

## Invariants

The tolerance decision is still made on physical separation. Center-pair orientation, Becke scalar graph, clamp order, log/log1p accumulation and normalization remain unchanged. The finite-reciprocal branch is not asserted bit-identical to the original division; its existing independent CUDA numerical gates remain mandatory. Future consumers must respect the signed geometry encoding rather than assume every nonzero entry is a reciprocal.

## Evidence

Seven host regression cases compile and execute the actual emitted geometry and partition bodies, including all five Becke iteration counts, equidistant points, ordinary/zero/subnormal separations and sentinel boundaries. Two cases fail before the repair and all seven pass afterward against the original quotient. Compiler dependencies came from a retained package; these are host arithmetic/indexing checks, not CUDA scheduling/libm or complete molecular-grid acceptance. No fresh GPU execution or endpoint speedup is claimed.

## Rejected alternatives

Rejecting valid zero/tiny tolerances would narrow the existing grid domain. Clipping the infinite reciprocal or the resulting NaN would change the physical quotient. Restoring all pointwise divisions would discard the optimization for ordinary geometries. The exceptional signed separation preserves the old equation only where its reciprocal representation fails.

## References

- #1197; #1100; #1117.
- [Original reciprocal-reuse decision](../performance/2026-09-24-cuda-quadrature-inverse-separation.md).
- `tests/python/test_quadrature_inverse_overflow.py`.

Agent: ChatGPT (Odd-PR Review)
Model: GPT-6 Astra Pro
