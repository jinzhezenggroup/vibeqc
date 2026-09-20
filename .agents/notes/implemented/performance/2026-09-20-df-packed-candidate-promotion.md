# Decision: stage generalized packed-response preferences before promotion

Status: implemented
Date: 2026-09-20

## Problem

PR #631 generalized a previously narrow production selector to every sm_120
workload with nbf squared times naux at least 2^28 and occupied rank at most
nbf/4. The existing small/large endpoint observations do not establish that
nearby unequal-auxiliary shapes or arbitrarily larger systems are faster.
Correct policy arithmetic is not complete-endpoint performance evidence.

## Decision

Keep the new CPU-safe workload preference and its policy tests as candidate
infrastructure. Do not call it from production auto-selection yet. Preserve the
previously qualified runtime selector byte-for-byte, including its device/domain
restriction, until the generalized promotion domain has measured evidence.
Explicit packed overrides remain available for endpoint qualification.

This stages the migration; it does not finish removing the legacy exact-domain
selector and does not close #444 or #459.

## Rejected alternatives

- Promote every shape beyond a threshold interpolated from two endpoints. This
  assumes smooth performance across representation, tiling and resource cliffs.
- Disable the previously qualified packed path altogether. That needlessly
  removes an established production optimization while fixing an over-promotion.
- Report CPU policy tests or CUDA compilation as GPU performance qualification.
  Neither measures the complete response/force endpoint.

## Invariants

- Unqualified shapes retain the pre-existing dense/symmetric fallback.
- The already-qualified runtime behavior and all correctness/resource gates stay
  unchanged.
- Generalized candidate predicates use overflow-safe workload arithmetic.
- Promotion requires an explicit update of the candidate-only regression gate.

## Evidence

The exact candidate header was compiled with C++20, -Wall -Wextra -Werror;
20,007 boundary/random cases, including size_t extremes and unknown architectures,
matched a Python arbitrary-precision predicate. This is policy evidence only.
`test_df_packed_promotion_boundary.py` protects the candidate/production boundary.
No new GPU endpoint timing or generalized performance qualification is claimed.

## Revisit when

Complete cold/warm/changed-geometry and relevant batch endpoint measurements cover
near-threshold, unequal-auxiliary and larger-size cliff cases on qualified devices,
with numerical parity, semantic work, memory and fallback evidence retained under
`docs/performance_engineering.md`.

## References

- #631, #444, #459 and the prior #381 endpoint boundary.
- `src/scf/df_derivative_policy.hpp`.
- `src/scf/cuda/df_gradient_bridge.cu`.
- `tests/python/test_df_derivative_policy.py`.

Agent: ChatGPT (VibeQC PR review)
Model: GPT-6 Astra Pro
