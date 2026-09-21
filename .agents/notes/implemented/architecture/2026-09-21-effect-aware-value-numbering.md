# Decision: Shared effect-aware value numbering

Status: implemented
Date: 2026-09-21

## Problem

TensorIR's historical exact CSE used a SHA-256 content address as the sole
representative lookup.  That was deterministic, but a digest match was treated
as sufficient proof of equivalence.  Integral scalar graphs separately used
Python structural interning, so the two compiler consumers had no shared
effect/collision contract.

TensorIR also distinguishes logical tensor semantics from debug/source index
names.  A GVN guard based on the complete `TensorSpec` object is too strict:
equivalent equations such as alpha-renamed einsums would fail to reuse values.

## Decision

Introduce `common.value_numbering.ValueNumberTable` as the backend-neutral
contract.  It reuses only operations explicitly marked `EffectKind.PURE`;
the default remains `OPAQUE`, reusing #673's effect model.

A lookup key is only a bucket key.  A caller-supplied semantic-equivalence
predicate must still approve reuse.  Distinct values that collide in the same
bucket remain distinct and are counted as rejected key collisions.

The same table also owns explicit copy propagation: once an adapter proves an
operation is an identity, the new SSA definition adopts the source value
number instead of constructing a second semantic value.

TensorIR keeps the existing `exact_cse` rewrite name for compatibility, but
its implementation now performs safe identity propagation plus value
numbering.  Semantic equality uses the logical TensorSpec payload, operation
attributes, and already-canonical input representatives.  Debug index names do
not participate.  Floating-point operand order does participate.  When #826
precision-execution provenance is present, its storage/compute/accumulation/
math-mode contract is part of the GVN bucket key, and rewritten bindings are
transported through the existing precision remap/validation path.

Integral `expr.Graph` uses the same table for construction-time interning.
Its nodes are immutable scalar algebra operations, so that adapter uses the
`number_exact_pure` fast path: purity and the complete structural `Node` key
are proven at the adapter boundary, avoiding generic effect/collision overhead
inside the scalar-DAG hot loop.

## Rejected alternatives

- Digest-only CSE was rejected because a digest collision must not authorize a
  merge without semantic verification.
- Complete TensorSpec object equality was rejected because source/debug index
  names are not logical equation semantics.
- A new GVN-specific effect enum was rejected; liveness and value numbering
  must share one effect contract.
- Commutative/reassociated floating arithmetic was rejected as an implicit GVN
  canonicalization.  IEEE-sensitive order changes require a separate numerical
  contract.

## Invariants

- Opaque, effectful, or alias-uncertain operations never merge by default.
- A bucket-key match is never sufficient evidence for value equivalence.
- Tensor dtype, logical domains/spins/symmetries, role and differentiability
  remain part of semantic identity.
- Debug/provenance naming remains separate from semantic identity.
- GVN does not reorder or reassociate floating-point arithmetic.
- Values with different precision-execution contracts never share a GVN
  bucket, while equal contracts remain eligible for reuse.
- Copy propagation is restricted to already-proven identity rewrites.

## Evidence

- `test_compiler_value_numbering.py`: effect barriers, rejected collisions,
  pure-value reuse, copy propagation, IEEE association negative test, and
  Integral Graph adoption.
- Final combined value-numbering/TensorIR/precision/Integral/SCF/XC/Maple/
  codegen gate after reconciling #826: 153 passed, 19 skipped.  Incompatible
  accumulation contracts remain distinct through every optimizer pass, while
  equal contracts remain mergeable.
- `tools/check_compiler_structure.py`: 262 compiler modules, zero dependency
  errors.
- Ruff lint/format and Astral ty checks pass for the changed clean modules.
- Same-machine WB97M-V second-derivative program build against current master
  `b982bfcd`: 47.46 ms median on master versus 47.93 ms with shared value
  numbering (about 1.0% overhead after the exact-pure fast path; the initial
  generic adapter was rejected at about 27% overhead).
- The WB97M-V expression hash remains
  `f1bbdc7424bf3bdcca6f965bc0386a97c72f72b791e2329ad060b85c09a8679b`
  with 570 nodes on both revisions.  Generated CUDA mathematical source is
  byte-identical after excluding `XC_IDENTITY`; that identity intentionally
  changes because its contract fingerprints modified generator sources.
- PBE TensorIR Fock construction plus optimization remains 6 -> 6 nodes and
  keeps logical hash
  `fe36bd19173efb7ee9a983cf202d5103806d2d0561195d5ce341707d9ee89c70`;
  measured medians against current master were 16.89 ms on master and
  17.17 ms after the change (about 1.7% overhead).

## Consequences

Tensor optimizer identity is versioned forward because the CSE stage now has
stronger legality checks and records compact value-numbering diagnostics.
Existing rewrite names remain stable.

Construction-time Integral Graph interning pays a small shared-table
bookkeeping cost; the measured WB97M-V builder overhead is about 1% on this
host after adding the exact-pure fast path.  Production compile-time evidence
should continue to track that overhead before broader adapters are added.

## Revisit when

Structured regions, mutable buffers, loads/stores, or aliasing enter a compiler
IR.  Those operations require explicit effect/alias proofs before they may use
the pure-value path.  Profitability-aware CSE/rematerialization also belongs
above this legality layer rather than weakening it.

## References

- #830
- #826
- #673
- #682
