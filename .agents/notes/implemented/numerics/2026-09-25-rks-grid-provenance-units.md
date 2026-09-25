# Decision: validate RKS grid ownership in dimensionless partition units

Status: implemented
Date: 2026-09-25

## Problem

The #1251 native RKS direction failed all six new CI cases before its CPKS numerical acceptance. The retained original JUnit report identifies the partition-weight reconstruction check, not a provider budget. Native nested hypot and generated scaled-norm/Becke arithmetic have ordinary last-bit differences. At a remote H2 grid point the atomic measure is about 306184; a partition difference near 1.1e-16 becomes a weighted difference near 3.4e-11, far exceeding the old fixed absolute mass tolerance while the partition remains consistent.

## Decision

Check the same 2e-14 absolute and 2e-13 relative constants on dimensionless ownership fractions: native weight divided by its retained atomic measure versus the generated partition value. Require finite real nonnegative measures and weights, fractions in [0,1], and exact zero native weight wherever the atomic measure is zero. Reject nonfinite ratios. This intentionally changes the units of the provenance tolerance; it is not a claim of an unchanged mass-space gate.

Actual integration continues using the original native grid weights. The analytic partition direction, XC expressions, AO motions, one-/two-electron derivatives, CPKS solve, molecular density/weighted-density finite-difference thresholds and translation tests are unchanged.

## Rejected alternatives

Do not raise provider budgets based on mismatched log excerpts, remove provenance validation, clamp partition values, replace the native integration grid, or relax the downstream response oracle. A fixed absolute weighted tolerance also allowed large relative provenance errors when atomic measures were tiny; normalization rejects that case.

## Evidence

Fourteen focused host checks pass, five fail against the old validation block. A 640-point H2 construction compares the shared generated partition with an independent scalar nested-hypot/Becke evaluation; maximum dimensionless difference is below 2e-14. Local execution isolates the checker and uses retained compiler dependencies, not a rebuilt native RKS library. Full current-head reconverged density and weighted-density tests remain mandatory and were not counted as passing in this repair.

## Revisit when

A native analytic grid-response owner supplies the identical rounded primal, or supported grid prescriptions admit signed atomic measures. Keep exact state/grid identities and independent response acceptance.

Refs #1251.

Agent: ChatGPT (Odd-PR Review)
Model: GPT-6 Astra Pro
