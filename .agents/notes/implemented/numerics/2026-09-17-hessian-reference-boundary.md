# Decision: keep the semi-numerical Hessian oracle separate

Status: implemented
Date: 2026-09-17

## Problem

PR #414 used finite-difference integral derivatives and a dense CPHF replica,
yet described them as completion of #180's analytic integration. Eager package
exports also made unrelated weight/RHS helpers require optional PySCF.

## Decision

Keep this useful implementation in the explicitly imported `reference` module.
It is an independent, bounded oracle, not the accepted #178/#179 consumer. Keep
analytic integration open. Validate against PySCF's analytic Hessian and three
finite-difference-of-gradient steps, with truthful fixture and source records.

## Rejected alternatives

Renaming finite differences as analytic derivatives would leave the generated
provider chain untested. Replacing the dense response with the shared solver in
this oracle would weaken its independence without completing integral wiring.

## Invariants

Use converged all-electron closed-shell Cartesian references, tight fixed SCF
settings, positive finite steps and a checked true response residual. Do not
force Hessian symmetry before measuring it. Do not claim that a reduced
occupied/virtual CPHF solve is inherently incorrect: its metric-density RHS
must include the known occupied connection, as specified by the shared helper.

## Evidence and consequences

`tests/python/test_hessian_reference.py` compares H2, custom 12-AO water, and
real 7-AO STO-3G water to analytic PySCF and all three gradient-difference steps.
The evidence driver retains basis/geometry, source hashes and explicit gates.
The implementation remains dense and intentionally bounded to 18 AOs/four
atoms; it adds no public method or GPU capability.

## Revisit when

The #178 weighted derivative providers and #179 shared response solve are wired
into the actual analytic assembly, with this oracle as an independent check.
