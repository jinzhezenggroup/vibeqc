# Decision: preserve the pinned SCAN scalar switching contract

Status: implemented
Date: 2026-09-20

## Problem

SCAN and SCAN0 extend the scalar scientific catalog with tau-dependent formulas.
Treating SCAN as rSCAN/r²SCAN or evaluating both singular interpolation branches
at alpha=1 would change the method or produce invalid intermediate values.

## Decision

Translate the pinned Libxc 7.0.0 SCAN exchange/correlation expressions into the
existing scalar DAG. Preserve the component-specific machine-epsilon switching
cutoffs and evaluate/differentiate only the selected branch. SCAN0 keeps its
one-quarter exact exchange as a separate typed primitive; the scalar evaluator
computes only its three-quarter exchange plus full correlation contribution.

## Rejected alternatives

Do not substitute r²SCAN regularization, introduce an arbitrary density floor,
fit an unversioned alpha smoothing polynomial, or add method-name-specific
runtime scientific branches. Do not infer public molecular KS, force, Hessian,
GPU schedule, or grid-tail qualification from the scalar catalog addition.

## Evidence

Retained PySCF 2.14.0 / Libxc 7.0.0 fixtures exercise both spin modes, typical and
boundary inputs, energy/first derivatives, and the default full packed feature
Hessian. Raw oracle arrays are retained alongside the structurally exact
spin-separability checks. Named SCAN/SCAN0 tests and alpha=0.99, 1, 1.03 tests
cover the switching neighborhood. Fresh combined SCAN/XC/MethodIR/PW91/Hessian
regression after master reconciliation: 165 passed without skips.

The registration-source integrity test exposed one missing final newline in
each newly pinned C file. Restoring the upstream bytes, rather than blessing new
hashes, preserved the original manifest and all scientific fixture values.

## Invariants and revisit conditions

No runtime PySCF/Libxc dependency is introduced. Unknown or unsupported domains
fail explicitly. A different cutoff, low-density/iso-orbital limit treatment,
public molecular endpoint, or production schedule requires independent numerical
acceptance and a corresponding versioned scientific/capability decision.

## References

- #611
- docs/xc_expressions.md
- python/vibeqc_compiler/xc/expressions.py
- tests/python/test_scan_family.py
- tests/python/test_xc_expressions.py
