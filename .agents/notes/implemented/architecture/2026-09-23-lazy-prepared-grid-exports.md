# Decision: activate prepared grid execution on explicit import

Status: implemented
Date: 2026-09-23

## Problem

Plain `import vibeqc` imports `GridSpec` from the compiler's DFT package. Python
first executes that package initializer, whose eager `PreparedGrid` re-export
loaded `prepared`, `cuda`, the native compiler adapters and JIT runtime. The
new import-activation regression therefore failed on its unchanged master base.

## Decision

Keep grid metadata and pure scientific definitions eagerly available. Resolve
`PreparedGrid` and `PreparedGridBatch` through module `__getattr__` only when
explicitly requested. Cache and return the canonical classes from `prepared`;
retain the public export list, discovery and static typing imports.

This changes activation, not packaging or scientific code. Removing the compiler
from the distribution would break intentional prepared/JIT consumers, and a
second proxy class would change class identity. Neither is necessary.

## Invariants and validation

The fresh-process regression forbids toolchain lookup and subprocess launch
during ordinary runtime import and checks that compiler adapters remain unloaded.
It then explicitly imports the prepared classes and checks canonical identity.
Existing prepared-grid CPU tests cover actual execution, geometry replacement,
resource admission and lifetime after activation. Compiler dependency checks
must continue to pass without GPU probing or production reference-oracle work.

## Revisit when

Another metadata re-export starts importing execution machinery. Preserve the
same explicit activation boundary rather than weakening the import regression.
