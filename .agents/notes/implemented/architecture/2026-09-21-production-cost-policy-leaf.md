# Decision: Isolate production compile-cost policy as a leaf module

Status: implemented
Date: 2026-09-21

## Problem

`integral/production.py` combined manifest/profile orchestration, source emission,
registry generation, bundle writing, and the stable AOT compile-cost/shard policy.
Issue #487 requires those responsibilities to acquire stable owners without
changing generated source identity or shard assignment.

## Decision

Move the compile-cost model, stable shard-slot table, triangular shell-class
index, and deterministic shard partitioner into `integral/production_cost.py`.
The new module exposes a read-only structural protocol instead of importing
`KernelSelection`, so dependency direction remains one-way: production
orchestration consumes cost policy, and cost policy does not import production.

`integral.production` keeps compatibility aliases for existing repository and
external imports during the migration. The package root now exports the cost
functions directly from their canonical leaf owner.

## Rejected alternatives

Creating `integral/production/cost.py` while retaining `integral/production.py`
would create an ambiguous module/package migration boundary. Importing
`KernelSelection` back from `production.py` into the new leaf would introduce the
cycle that #487 explicitly aims to avoid. Recomputing shard placement during the
move was also rejected because stable shard identity is a build/cache contract.

## Invariants

- Stable AOT shard-map version and slot assignments do not change in this move.
- Production shard/profile source text and bundle bytes remain deterministic and
  byte-identical for the same manifest and target.
- Existing imports from `vibeqc_compiler.integral.production` continue to resolve
  to the canonical cost-policy objects.
- The cost-policy leaf must not depend on production emission, benchmark, CLI, or
  runtime concerns.

## Evidence

Against parent `c4bbadd12` for the checked-in `sm_120` production manifest:

- all-selection shard SHA-256: `aa5c0bfb2eb6c16394d7c58e6042e96a0df604c627caf5849040fc71517f582e` in both trees;
- all-selection profile shard SHA-256: `0a53a5ef54bd83f422bf4cbc95b09752fe614193a2ff567e711e7b334b918356` in both trees;
- 8-way production bundle aggregate SHA-256: `6760610daeb36d7048b6f1612aa0bf25cd0fec75729dcfd0b0e25702d276d12c` in both trees;
- focused compiler/codegen suite: 355 passed, 48 skipped;
- compiler structure audit: 257 modules, 0 dependency errors.

## Consequences

This is one ownership slice of #487 rather than the final package decomposition.
`production.py` remains the compatibility/orchestration module for selection,
emission, registry, and bundle responsibilities until follow-up slices move those
owners. New compile-cost or stable-shard policy belongs in `production_cost.py`.

## Revisit when

The remaining production orchestration has been decomposed far enough to replace
`production.py` with a `production/` package. At that point this leaf can move to
`production/cost.py` with the same one-way dependency and compatibility rules.

## References

- #487
- Parent snapshot: `c4bbadd12`
