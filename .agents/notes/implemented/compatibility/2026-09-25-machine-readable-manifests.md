# Decision: keep non-rendered repository inventories under manifests

Status: implemented
Date: 2026-09-25

## Problem

The documentation root mixed reader-facing pages and rendered generated
reference material with JSON files used only by repository checkers, maintenance
inventories, and test fixtures. That made `docs/` look like the ownership home
for machine contracts and encouraged one-time engineering state to accumulate
beside the public documentation.

The earlier audience-documentation migration correctly retained paths that are
rendered, generated into documentation, hashed by scientific metadata, or
otherwise part of a compatibility contract. It did not require every unrelated
checker input to remain under `docs/`.

## Decision

Keep reader-facing current-state documentation under `docs/`. Keep generated
or compatibility-sensitive documentation artifacts at their existing documented
paths when producers or serialized identities require those paths.

Place machine-readable repository contracts that are not rendered documentation
under `manifests/`. Use `manifests/maintenance/` for maintenance/checker
inventories, while scientific/generated public manifests and fixtures may live
at the manifests root or their existing subdirectories.

This change moves the compiler optimization ledger, Direct-JK tuning inventory,
electronic-structure production-path inventory, periodic asset inventory, CUDA
ownership baseline, and IntegralIR example fixture without changing their bytes.
Their checkers, tests, and current reproduction commands consume the new paths.

## Invariants

- Do not move generated Libxc reports, the public method table, CUDA ownership
  rendered ledger, density-source forwarding page, or XC-expression forwarding
  page merely for directory neatness; their producers, readers, hashes, or
  serialized identities own those paths.
- Historical benchmark/provenance artifacts retain the paths they recorded at
  measurement time.
- A relocation must update every current producer, checker, test, and
  reproduction command in the same change.
- `docs/` remains current-state and audience-oriented rather than a general
  storage directory for JSON state.

## Consequences

Repository inventories now have a predictable machine-oriented home, while the
documentation root contains only reader-facing pages, build files, forwarding
compatibility pages, and generated material that is intentionally rendered or
path-sensitive.

## References

- [Audience documentation path decision](2026-09-23-audience-documentation-paths.md)
- `docs/AGENTS.md`
- `docs/index.md`
- `manifests/README.md`
