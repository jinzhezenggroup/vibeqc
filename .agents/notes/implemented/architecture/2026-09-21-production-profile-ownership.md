# Decision: Separate production manifest/profile ownership from emission orchestration

Status: implemented
Date: 2026-09-21

## Problem

`integral.production` still owned production-manifest schema parsing, target/profile
resolution, schedule-row validation, source emission, registry serialization, and
bundle writing after the cost and selection-policy slices of #487.  Manifest
resolution is needed by capability reporting and other compiler consumers that do
not need to import the full CUDA emitter, so leaving it in the orchestration
module preserved an unnecessary dependency on emission code.

## Decision

Move `ProfileMatch`, `ResolvedProductionProfile`, manifest/profile resolution,
row validation, and the `load_production_*` compatibility views into
`integral.production_profile`.  `integral.production` remains the migration
facade and re-exports the canonical public objects plus the repository-used
`_schedule_from_payload` helper by object identity.

Capability reporting now lazy-imports `resolve_production_profile` from the
canonical profile owner rather than from the emission facade.  The lazy import is
retained because profile resolution consumes capability normalization; executing
the import only after module initialization avoids a module-import cycle.

The existing `ty` warning override moves from `production.py` to
`production_profile.py`: the dynamic JSON-dictionary diagnostics moved with the
unchanged parser, while the remaining emission module returns to normal strict
checking.

## Rejected alternatives

Duplicating profile/result types in a compatibility module was rejected because
callers must observe one canonical class/function identity.  Migrating every
repository caller in the same change was rejected because #487 explicitly uses
narrow compatibility re-exports to keep ownership slices independently
reviewable.  Moving source emission or registry serialization with profile
parsing was also rejected because those responsibilities change for different
reasons and are subsequent #487 slices.

## Invariants

- Legacy imports from `integral.production` resolve to the canonical
  `integral.production_profile` objects.
- Manifest schema behavior, profile fallback/compatibility semantics, schedule
  validation, generator ABI checks, and target validation are unchanged.
- Generated production/profile shards and registry files remain byte-identical
  for the same manifest and target.
- Profile ownership must not depend on production emission, bundle/cost owners,
  benchmark packages, CLI tools, or the public runtime.
- Selection policy remains canonical in `production_selection`; this module
  consumes it rather than defining a second selection abstraction.

## Evidence

- AST comparison against base `391de66c1` found all 17 moved class/function
  definitions structurally identical.
- Focused compiler/codegen regression: 295 passed, 48 skipped.
- Additional manifest/profile selection regression: 100 passed.
- Compiler structure audit: 261 modules, 0 dependency errors.
- `ty` exits successfully after transferring the existing warning override to
  `production_profile.py`; `production.py` is no longer in that override.
- Base/current generation with the checked-in `sm_120` manifest and eight stable
  shards produced 10/10 byte-identical files for both single-profile and
  namespaced multi-profile bundles. Aggregate SHA-256 values were
  `bb3f2d759543aa50ec364de8f0861d38665b911609554fe42244342e451e69eb` and
  `79163aaaf72bb92e6838ad34d936c2f8be48f03b66925dae2318a2ba38a1d48f`.

## Consequences

Profile resolution can now be reused without importing CUDA source emission, and
`production.py` is smaller and more narrowly focused on emission/registry/bundle
orchestration.  Registry serialization, source emission, and filesystem/bundle
ownership remain to be split in later #487 slices.

## Revisit when

Remove the compatibility re-exports only after repository and downstream callers
no longer import profile objects/helpers from `integral.production`.  Revisit the
`ty` warning override independently when the manifest JSON typing debt is paid
down.

## References

- Issue #487
- `python/vibeqc_compiler/integral/production_profile.py`
- `python/vibeqc_compiler/integral/production.py`
- `tests/python/test_production_profile.py`
