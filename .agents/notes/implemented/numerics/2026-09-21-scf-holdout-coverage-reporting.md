# Decision: expose force-envelope coverage in SCF holdout reports

Status: implemented
Date: 2026-09-21

## Problem

The NUM03 SCF-effort estimator reported false-success and overconservative
counts, but it did not expose whether its empirical max-force envelope covered
the independently observed strict error. A broad target budget can therefore
show zero false-success while the predicted force envelope still underestimates
the measured force error.

## Decision

`ScfForceErrorEstimator.evaluate_holdout` now reports the existing pass/fail
fields together with:

- per row: `force_covered`, `force_underestimated`, and
  `force_underestimation_factor` (`actual / predicted`);
- aggregate: `force_coverage`, `force_coverage_rate`,
  `force_underestimation_rows`, `max_force_underestimation_factor`, and
  `force_underestimation_nonfinite_rows`.

The aggregate maximum considers underestimation rows only. A zero predicted
force with a nonzero observed force, or a ratio that exceeds floating-point
reporting range, contributes to `force_underestimation_nonfinite_rows`.
If such a row exists, the maximum is `None` instead of silently omitting the
largest miss; its row ratio is also `None`, keeping strict JSON valid. Zero
over zero is defined as 1.0 and is covered.

The estimator's admission domain, thresholds, and strict fallback behavior are
unchanged. Coverage is evidence about an empirical envelope, never a certified
bound.

## Invariants

1. Existing report keys and policy decisions remain backward compatible.
2. Coverage compares the predicted and independently observed max-force errors;
   it does not use the target budget as a proxy.
3. A force-envelope miss remains evidence only and cannot relax strict fallback.
4. Empty holdout input returns zero coverage rates and no maximum factor.

## Evidence

`tests/python/test_force_aware_scf.py` covers a mixed covered/missed holdout
under a broad target, per-row ratios, aggregate counts, a zero-envelope miss,
and empty holdout reporting.

## References

- GitHub issue #175 (NUM03)
- `python/vibeqc/force_aware_scf.py`
- `tests/python/test_force_aware_scf.py`
