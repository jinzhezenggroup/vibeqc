# Decision: xc potential geometry jvp

Status: implemented
Date: 2026-09-22

## Decision

Expose the semilocal potential geometry directional derivative and assemble directional coefficients through existing AO/grid response owners. Only the missing JVP slice is added; merged mixed-geometry infrastructure remains canonical.

## Invariants and rejected alternatives

Keep moving-grid, AO and coefficient contributions consistent with the same displaced geometry. Do not promote this primitive into a complete public Hessian capability.

## Evidence and remaining qualification

The XC contraction suite passes against the current CPU AO provider (46 tests), including finite-difference comparisons. Complete nuclear-response endpoint qualification remains separate.

## Revisit when

Revisit the candidate when the stated endpoint gates pass or the shared owner changes.
