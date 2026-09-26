# Decision: apply numerical acceptance to every measured repeat

Status: implemented
Date: 2026-09-17

## Problem

The batch comparator published every paired energy/force error but gated only
iteration-matched pairs when available, falling back to the last pair otherwise.
An earlier inaccurate sample could therefore enter the ordinary timing median
without failing numerical acceptance. Matching iteration counts does not prove
that either result meets the requested accuracy.

The #206 AOT follow-up independently checked every pair and retained the complete
96-AO failures. Its original summary reports only the final-pair error, while
its retained raw arrays reveal larger errors in earlier repeats. Those historical
bytes remain unchanged.

## Decision and invariants

The gate now uses maximum energy and requested-force errors across all measured
pairs. Record `selection: all_measured_pairs` and the complete pair count so
consumers can distinguish the strengthened acceptance from historical records.
Preserve all fixture-specific error tolerances, SCF controls, raw samples and
iteration-matched timing diagnostics. No-repetition input cannot pass acceptance.
Energy-only force error remains absent, not a measured zero.

Also clarify that legacy integral/contraction timing fields hold whole endpoint
durations. Cold timing contains preparation, SCF and requested properties; warm
timing contains the resident-plan solve. Neither is a measured component split.

## Validation and limits

Regression cases place an inaccurate pair before a passing final pair, both
with and without an iteration-matched pair. Both energy and force gates must
reject them. Existing energy-only, finite-value and gate-command tests preserve
their contracts. The four saved AOT points can be re-evaluated offline without
rerunning or relabeling GPU timing.

This change prevents selective acceptance; it does not repair the scientific
force failures or qualify previously failed performance. #206 remains open.

## References

[#423 retained AOT matrix](https://github.com/jinzhezenggroup/vibeqc/pull/423),
[#422 binary provenance](https://github.com/jinzhezenggroup/vibeqc/pull/422), and
[performance qualification policy](../../../../docs/maintainer/performance_engineering.md).
