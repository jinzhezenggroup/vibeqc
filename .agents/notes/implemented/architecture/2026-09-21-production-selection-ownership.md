# Decision: Separate production selection policy from emission orchestration

Status: implemented
Date: 2026-09-21

## Problem

`integral.production` owned both the scientific validity contract for a selected
kernel and the much larger manifest, source-emission, registry, and bundle
orchestration surface.  That coupling made selection/capability policy difficult
to reuse or test without importing the full production emitter, and it worked
against the ownership split tracked by #487.

## Decision

Move `KernelSelection`, recurrence capability checks, and `_selection_integral`
into `integral.production_selection`.  The package-level `KernelSelection` export
now comes directly from that module.  `integral.production` keeps compatibility
re-exports of the exact same objects so downstream imports do not need to move in
lockstep.

The extraction is intentionally behavior-neutral: manifest parsing, profile
resolution, source emission, registry generation, bundle layout, and compile-cost
policy remain unchanged.

## Rejected alternatives

Keeping a duplicate compatibility class in `production.py` was rejected because
it would create two nominally equivalent selection types and violate the compiler
compatibility rule that facades forward canonical objects.  Moving profile
resolution at the same time was also rejected to keep this slice independently
reviewable and to avoid mixing manifest parsing with selection ownership.

## Invariants

- `integral.production.KernelSelection` and
  `integral.production_selection.KernelSelection` are the same class object.
- Existing recurrence legality and CUDA mapping validation remain unchanged.
- Generated source, shard layout, registry bytes, and artifact identity do not
  change because of this ownership move.
- Selection policy must not import production emission/bundle orchestration.

## Evidence

- `tests/python/test_production_selection.py` checks compatibility object identity
  and the leaf dependency direction.
- Production recurrence, generated-kernel signature, and compile-cost tests pass
  unchanged after the move.
- Generating the accepted `sm_120` production manifest with eight stable shards
  before and after the extraction produced 10 files with byte-identical SHA-256
  digests (eight shards plus registry source/header).

## Consequences

Future #487 work can move profile, emission, registry, and bundle responsibilities
behind narrower modules without re-owning scientific selection validation.  The
legacy `integral.production` import remains supported while callers migrate.

## Revisit when

Revisit the compatibility re-export only after repository and downstream callers
no longer import `KernelSelection` from `integral.production`.

## References

- Issue #487
- `python/vibeqc_compiler/integral/production_selection.py`
- `tests/python/test_production_selection.py`
