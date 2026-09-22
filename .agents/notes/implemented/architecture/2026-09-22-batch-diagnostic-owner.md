# Decision: Own batch diagnostics outside the execution facade

Status: implemented
Date: 2026-09-22

## Problem

`vibeqc.batch` owned both prepared execution and the growing set of native
diagnostic/profile records and translations. Adding backend observability
therefore expanded the execution facade even when execution semantics did not
change.

## Decision

`vibeqc._batch_diagnostics` owns batch diagnostic records, pure native-record
decoders, and the narrow diagnostic read transactions. `PreparedBatch` keeps the
public `last_*` methods, open/opt-in guards, and delegates each read to that
owner. The dependency direction is `_batch_diagnostics -> _native`; the internal
owner must not import `batch` or `calculator`.

`vibeqc.batch` and the package root continue to re-export the same class objects.
The classes retain `vibeqc.batch` as their pickle global path so new payloads and
payloads written before the extraction use the established public facade.

## Rejected alternatives

- Moving only decoder functions would leave record growth owned by the facade.
- Giving diagnostics an execution context would create a second broad owner for
  the native handle and its lifetime.
- Exposing the internal module as a replacement public path would change type
  and serialization identity for an ownership-only refactor.

## Invariants

- Public imports, dataclass fields/defaults, derived properties, and JSON-ready
  payloads remain compatible.
- Fixed-size profile reads make one native call; variable-size reads retain the
  existing query-plus-copy sequence and changed-count errors.
- Native status translation remains `_native.check`; opt-in guards run before a
  reader is invoked.
- Diagnostic ownership does not alter prepared execution, warm starts, resource
  accounting, native ABI, or scientific calculations.

## Evidence

- Pure decoder, enum, empty-result, status/error, call-count, dependency, import,
  and pickle compatibility tests in `tests/python/test_batch_diagnostics.py`.
- A direct `origin/master` before/branch-after fixture comparison covers all five
  `PreparedBatch.last_*` values and their native call sequence.

## Consequences

New batch diagnostic records and translations have a focused owner, while the
facade preserves its existing API. The explicit pickle-path assignment is a
small compatibility cost that should remain until a separately versioned public
serialization change is approved.

## Revisit when

Revisit the public serialization path only with a versioned migration, or split
the internal owner further if diagnostics acquire independent native lifetimes.

## References

- Issue #490.
