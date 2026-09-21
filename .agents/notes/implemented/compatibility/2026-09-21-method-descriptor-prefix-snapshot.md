# Decision: materialize bounded method-descriptor prefixes at the C boundary

Status: implemented
Date: 2026-09-21

## Problem

Review of #903 found an optional precision read outside its extent guard. Real
protected-page tests then exposed a second problem with Clang 18 at `-O3`:
interprocedural argument promotion of `validate_option_family` hoisted the
`ks_options` load into `prepare_calculation`, before the callee's size guard.
The same guarded test passed with GCC Release and Clang `-O2`; optimization
level must not determine whether a legitimate legacy C caller is memory-safe.

## Decision

After validating the mandatory C header/prefix, copy at most the caller's
reported extent into a zero-initialized current-sized method descriptor, then
pass that complete object to C++ method preparation. Share the implementation
between single and batch C entrypoints. Keep the caller's original `struct_size`:
missing options must retain their version-specific defaults, not become explicit
zero-valued suffix options. Nested pointees remain borrowed during preparation;
existing scientific owners still snapshot their content before returning.

## Rejected alternatives

- Relying only on the optimized-away precision load or GCC tests misses Clang's
  argument-promotion path.
- Marking one callee noinline/volatile depends on optimizer behavior and does not
  establish a complete object at the language boundary.
- Enlarging struct_size would silently change absent-field semantics, including
  optional controls for which an explicit zero is not the historical default.
- Removing the protected-page regression would conceal a real compatibility bug.

## Invariants and evidence

Protected-page cases exercise two physically short method prefixes and all five
KS-option prefixes through both actual public preparation entrypoints. GCC and
Clang builds must both pass. The precision predicate continues to use the already
size-gated/defaulted `ScfOptions` value; this snapshot is not permission to ignore
semantic extent checks.

Agent: ChatGPT
Model: GPT-6 Astra Pro
