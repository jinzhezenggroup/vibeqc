# Decision: Split production emission from bundle writing

Status: implemented
Date: 2026-09-22

## Problem

`integral.production` still owned deterministic CUDA source emission and
filesystem publication after profile, selection, registry, and compile-cost
policy had moved to dedicated modules. Those responsibilities have different
dependencies and change for different reasons.

## Decision

Move the remaining source-emission and filesystem/bundle responsibilities out
of `integral.production`, retaining that module as an identity-preserving
compatibility facade.

The dependency direction is now:

```text
production (compatibility facade)
  -> production_bundle -> production_emission
  -> production_registry / production_profile / production_selection / production_cost
production_bundle
  -> production_emission
  -> production_registry / production_profile / production_cost / shell_spec
production_emission
  -> CUDA emission/lowering + production_registry / production_selection
  -> production_profile / production_cost / shell_spec
```

Leaf policy owners do not import the compatibility facade. `production_emission` owns deterministic CUDA source text and has no filesystem writes; `production_bundle` owns directories, stable shard layout, file replacement, and registry artifact publication. Registry serialization, profile parsing, selection policy, and compile-cost partitioning remain in their previously split owners.

Compatibility imports from `integral.production` retain object identity, including
the historical domain symbols that were exposed by its old imports, so existing
build tools and downstream callers do not need an eager migration. The
compatibility facade contains no source-generation or filesystem implementation.

## Rejected alternatives

- A mechanical package conversion would add import churn without improving the
  dependency direction.
- Reimplementing selection or registry policy in the new owners would fork the
  existing canonical objects and generated identity contracts.
- Removing the facade immediately would break repository and downstream imports
  before their migration is coordinated.

## Invariants

- Emission remains a pure source-text operation with no filesystem writes.
- Bundle writing owns directories, application of the stable shard layout, and
  artifact publication; `production_cost` owns slot assignment and partition policy.
- The compatibility facade forwards canonical objects rather than wrapping or
  duplicating implementations.
- Generated source, registry text, profile hashes, and stable AOT shard assignment
  remain byte-identical for a behavior-neutral ownership move.
- Production ownership imports remain cycle-free, and leaf modules never import
  the compatibility facade.

## Evidence

- All 16 function bodies moved from the pre-split module are AST-identical.
- Exact-base generation comparisons match all 10 single-profile and all 18
  multi-profile files by relative path, size, and SHA-256.
- The pinned legacy-artifact regression passes on Linux and verifies generated
  registry, shard, and manifest bytes.
- The focused Linux compiler suite passes with 97 tests and 2 optional skips;
  the compiler structure audit reports 313 modules and zero dependency errors.
- A Linux CPU CMake configure and `vibeqc` build complete through all generated
  source entry points.

## Consequences

New kernel, wrapper, and profile-shard source work has one emission owner, while
filesystem layout and publication changes have one bundle owner. This decision
records completion of the emission/bundle ownership slice; the facade is retained
until its remaining callers migrate.

## Revisit when

Remove a facade export only after repository and supported downstream callers no
longer import it and identity tests are the sole remaining consumer.

## References

- Issue #487
- PR #1024
