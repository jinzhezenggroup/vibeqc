# Decision: separate specialization metadata from execution and promotion

Status: implemented (contract only; production consumers are not migrated)
Date: 2026-09-19

## Problem

Issue #459 needs a common contract before DF selection (#444) and benchmark
qualification (#454) can converge. Moving endpoint-specific branches into a
new generic module would preserve the wrong abstraction. Reimplementing AOT or
user-local caches would create incompatible identity and safety owners.

## Decision

Add pure records and selection in `vibeqc_compiler.common.specialization`.
Reuse `TargetInfo` and `common.provenance.canonical_hash`; reference the existing
artifact, scientific, compiler, schedule and tuning-profile identities. The
compiler hash's existing owner must cover source/IR/generator/ABI/toolchain and
options. Capability projection is not an alternative binary-compatibility check.

Correctness guards and qualified performance domains are independent. Missing
promotion is conservative; the first eligible promoted profile wins in explicit
caller order. Fallback is separately correctness/identity checked, never implied
by failure of the optimized candidate. Unknown facts fail required predicates.

The records accept only immutable finite scalar facts. Equality and hashing of
scalar-bearing records preserve JSON types, including True versus 1 versus 1.0.
Guard conjunctions and input feature order are canonicalized. Candidate priority
is deliberately not sorted: changing priority changes the selected implementation.

## Rejected alternatives

- No endpoint or product-name tables in generic code. Consumers provide general
  features and qualified domains, with exact measured identities in provenance.
- No callable/eval predicates: simple equality and inclusive bounds are portable
  to future native adapters and make missing-fact behavior explicit.
- No binary loading, probing, compilation, new cache directories or replacement
  artifact-key recipe. The existing loader is still the final reuse authority.
- No production promotion in this slice. CPU synthetic guards cannot establish
  real-device endpoint parity or performance.

## Evidence and boundaries

`PYTHONPATH=python python -m pytest -q tests/python/test_specialization.py`
checks generalized shapes, inclusive bounds, missing capabilities, unpromoted
profiles, stale identities, safe fallback, strict scalar types, immutability,
priority, deterministic identities and original cache-key/loader preservation.
Existing compiler structure tests enforce the one-way `common` dependency.

No generated equations, numerical thresholds, production selectors or existing
loader/cache formats are edited. Adding a compiler leaf naturally affects the
existing complete source inventory; that conservative invalidation is preserved,
not hidden by freezing an obsolete source hash.

## Revisit when

Migrate native DF selectors and benchmark capability checks using this contract,
without a Python callback per kernel or copied policy. Add predicates only for a
concrete consumer need. Retain each production numerical/endpoint gate and audit
unknown-device/workload fallback separately. Tensor/DFT/local-profile adoption is
follow-on work, not a dependency of unrelated scientific correctness changes.

Refs #459, #444, #454, #136, #168.

Agent: ChatGPT
Model: GPT-6 Astra Pro
