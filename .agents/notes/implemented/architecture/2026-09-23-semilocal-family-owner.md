# Decision: retire the legacy semilocal expression module

Status: implemented
Date: 2026-09-23

## Problem

The semilocal family composer had already been separated from production-domain
continuations and shared native lowering. Its legacy `expressions.py` name still
required an import allowance and obscured completion of that ownership split.

## Decision

Move the current family composer byte-for-byte to `xc/semilocal_family.py` and
physically delete `xc/expressions.py`. Migrate the dispatcher, geometry lowering,
and qualification imports. Remove all legacy semilocal consumer allowances.
Retain the independently scoped RSH retirement ceiling. Preserve the current
`production_policy.py` and shared `semilocal_codegen.py` owners during integration;
tool entry points must not regain differentiation or continuation mathematics.

Update FunctionalSpec provenance and the source registry to the actual new owner.
The path change intentionally changes source-derived artifact identities while
preserving mathematical expressions, constants, and ABI layout.

## Rejected alternatives

A forwarding legacy module would keep the obsolete ownership surface alive.
Reusing the old branch versions of production policy or tool root construction
would undo the intervening shared-lowering and PW92 source-ownership work.

## Invariants

The family composer remains compiler-only and has no native runtime dependency.
Retired imports fail the structural gate; they do not silently regain permission.
Family composition and numerical continuation have separate canonical owners.

## Evidence

The integrated composer is byte-identical to the preceding main branch composer.
Regenerated complete CPU XC and CUDA r2SCAN headers are byte-identical to the
main baseline after replacing only their 64-digit provenance hashes. The compiler
structure check inspects 320 modules with zero dependency errors. Relevant
retirement, importer, generation, and provenance tests cover the moved consumers.

## Revisit when

Further XC family retirement changes the remaining RSH dispatcher boundary, or a
new family requires a different compiler-owned composition interface.

## References

PR #1048; the shared-lowering decision is recorded in
`2026-09-23-dft-shared-semilocal-cpu-cuda.md`.
