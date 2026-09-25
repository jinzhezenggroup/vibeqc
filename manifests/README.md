# Repository manifests

This directory contains machine-readable repository contracts and inventories.
They are consumed by generators, validation tools, and CI rather than rendered
as user documentation.

- Top-level manifests describe public/generated scientific inputs or fixtures.
- `maintenance/` contains current repository-state inventories used by
  fail-closed maintenance checks.

Historical rationale belongs in `.agents/notes/`; prose describing current
behavior belongs in `docs/`.
