# Decision: Separate the Python calculation facade owners

Status: implemented
Date: 2026-09-27

## Problem

`Calculator` and `PreparedBatch` combined stable public records and orchestration
with basis/model identity, native result decoding, backend labels, and warm
restart bookkeeping. Every new diagnostic or restart feature therefore enlarged
the public execution modules and made pure API behavior difficult to test.

## Decision

The public records (`Atom`, `Primitive`, `Shell`, `Result`,
`CorrelationResult`, and `MethodCapabilities`) live in `_api_types`. Bundled and
explicit basis snapshots plus native-free `ResolvedModel` construction live in
`_model_resolution`. Native descriptor/status/force/correlation decoding lives
in `_result_translation`. Python-side warm-start, checkpoint, projection, and
restart metadata live in `_warm_state`.

`calculator.py` and `batch.py` retain the existing public and compatibility
imports while delegating to those owners. Checkpoint and progressive helpers
access the warm-state owner directly; compatibility properties remain only for
older internal callers. Native handles and device-resident densities remain
owned by the existing native batch/calculation objects.

## Rejected alternatives

- A universal context object was rejected because it would recreate the coupling
  the extraction is meant to remove.
- Changing public record locations or removing legacy imports was rejected to
  preserve pickle paths and downstream imports.
- Moving native execution itself into the new modules was rejected because this
  slice is behavior-neutral and must not duplicate provider/backend ownership.

## Invariants

- `Calculator` and `PreparedBatch` public behavior, status/error semantics,
  cache identity, and prepared-state invalidation remain unchanged.
- Model resolution does not construct a native execution context.
- Result translation does not import either execution facade.
- Warm metadata and restart/projection sets have one Python owner and are reset
  whenever native warm starts are cleared.
- Unsupported scientific/backend combinations continue to fail closed.

## Evidence

- `tests/python/test_api_ownership_boundaries.py` covers record identity,
  native-free model identity, result translation, warm-state lifecycle, and
  one-way dependency direction.
- The pure boundary suite passes with `PYTHONPATH=python:$PWD python3 -m pytest
  tests/python/test_api_ownership_boundaries.py -q`.

## Consequences

The facade modules are smaller and future diagnostic/result/state additions have
explicit owners. The compatibility properties add a small amount of forwarding
code, but they keep checkpoint/progressive integrations source-compatible while
the migration completes.

## Revisit when

Remove compatibility forwarding only after all repository consumers use
`WarmStartState` directly and a deprecation window is documented.

## References

- Issue #490: refactor Python calculation facade.
- PR #1034: extracted batch diagnostics, the preceding slice of #490.
