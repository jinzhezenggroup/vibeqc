# Decision: separate B3LYP endpoint policy from legacy RSH mathematics

Status: implemented
Date: 2026-09-23

## Problem

The extended-family builder mixed audited Maple components with retained B3LYP
endpoint continuations and a duplicate handwritten PW92 correlation expression.
This concealed the remaining production policy during legacy module retirement.

## Decision

Move the existing full-range B88 and LYP production continuations into
`xc/b3lyp_production_policy.py`. Keep the ordinary imported interior components,
range-separated ITYH mathematics, and composition-level vacuum policy under their
existing owners. Delete unreachable handwritten short-range B88 alternatives and
route the extended-family LDA_C_PW component through the canonical pinned PW adapter.

Keep the shared compiler native lowerer and the physically retired semilocal
legacy module from the integrated parent. Update the source registry to the real
policy owners; source-derived identities intentionally reflect the move.

## Rejected alternatives

Deleting the endpoint continuations would reintroduce zero-gradient B88 derivative
and zero-spin LYP singularities. Renaming the entire legacy module would retain
obsolete duplicate mathematics. Restoring tool-owned root construction while
merging the older branch would undo the shared compiler lowerer.

## Invariants

B88 keeps its analytic x*asinh(x) series before differentiation and lazy zero-spin
branch. LYP keeps cancellation of apparent negative spin-density powers. The
B3LYP-only production admission and 1e-18 total-density vacuum policy are unchanged.
The RSH/PW consumer must agree with an independent Libxc reference through fxc.

## Evidence

The complete regenerated CPU XC header is byte-identical to the integrated
parent after replacing only provenance hashes, including all B3LYP production
arithmetic, constants, and ABI. The extended-family PW replacement is tested
against the pinned Libxc 7.0.0 C-API energy, gradient, and Hessian fixtures in both
spin layouts at 2e-12 relative plus 2e-13 absolute tolerance. Existing LYP/PW91,
RSH, source-registry, and retirement checks qualify the surrounding consumers.

## Revisit when

A generic imported production-domain continuation replaces these endpoint
policies with independent molecular and derivative qualification.

## References

PR #1050 and the parent retirement in PR #1048.
