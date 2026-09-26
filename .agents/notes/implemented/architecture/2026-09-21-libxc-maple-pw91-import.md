# Decision: admit PW91 through the common Libxc Maple frontend

Status: implemented
Date: 2026-09-21

## Problem

Issue #743 extends the pinned-source importer beyond PBE and SCAN/r2SCAN.
PW91 exchange uses Maple `arcsinh`, while PW91 correlation composes the
already-qualified PW correlation include graph. Keeping a handwritten-only
PW91 path would preserve the duplicate source-of-mathematics problem that
#739 is intended to remove.

## Decision

Admit only the source-evidenced unary `arcsinh(x)` construct and lower it
directly to the existing Graph `asinh` operation. Bump importer semantic
identity to `libxc-maple-graph/v5`. Qualify both `GGA_X_PW91` and
`GGA_C_PW91` through energy, first feature derivatives, packed feature
Hessians, and both Scalar C and CUDA emitters.

This change does not cut production construction over to the imported graph;
that remains owned by #744.

## Rejected alternatives

- Do not add a generic Maple transcendental dispatcher. Each admitted
  construct remains explicit and fail-closed.
- Do not special-case PW91 formulas in a new Python translation. That would
  recreate the duplicate mathematical source this migration is removing.
- Do not combine production cutover with source qualification.

## Invariants

- Pinned Libxc sources and transitive hashes remain provenance inputs.
- Graph AD owns derivatives; no external symbolic runtime is introduced.
- Existing VibeQC feature/spin conventions remain the adapter boundary.
- Independent Libxc 7.0.0 Hessian fixtures remain an acceptance oracle.

## Evidence

`tests/python/test_libxc_maple_pw91.py` checks imported-vs-handwritten parity,
independent retained Libxc 7.0.0 E/vxc/fxc fixtures for both spin layouts, and
Scalar C/CUDA lowering.

## Consequences

The common frontend now covers one additional complete exchange/correlation
family with one narrowly admitted syntax extension. Artifact identities
intentionally change because importer semantics changed from v4 to v5.

## Revisit when

Only extend the frontend again when another pinned Libxc source requires a
construct that cannot be represented by the currently admitted subset.

## References

- #739
- #743
- #744

Agent: ChatGPT
Model: GPT-5.6 Sol
