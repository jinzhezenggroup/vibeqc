# Decision: make `vibeqc_compiler` the canonical scientific compiler package

Status: implemented
Date: 2026-09-15

## Problem

Compiler functionality had grown across large lowering/tuning modules and former
`tools/` package paths. Continued growth risked blurred ownership, duplicate
runtime/compiler dependencies, fragile imports, and future refactors that changed
generated artifacts accidentally.

The #239 decomposition also needed a durable record of what was intentionally
moved versus what scientific behavior was required to remain unchanged.

## Decision

Make `python/vibeqc_compiler` the importable canonical scientific compilation
subsystem. It owns mathematical IR/lowering, schedules and planning, generated
artifacts, hashes/resources/evidence, and finite compiler processes. The public
`vibeqc` runtime remains separate, while `src/integrals`, `src/tensor`, and
`src/dft` retain native interfaces, runtime allocation, and execution templates.

Split integral lowering and tuning into focused leaves behind stable facades.
Keep narrow compatibility modules under the former `tools/vibeqc_*` paths that
forward to canonical module objects instead of loading duplicate module trees.
Use the same logical compiler source inventory in checkout and installed use.

## Rejected alternatives

- Keep the former `tools/` namespaces as the canonical compiler implementation:
  this leaves compiler ownership coupled to command-line/repository layout rather
  than an installable subsystem.
- Let source generation import the public runtime, load native libraries, probe a
  GPU, or import PySCF/Torch/CuPy: generation must remain finite, deterministic,
  and usable from an uninstalled checkout with its documented dependencies.
- Duplicate scalar/scientific algebra for DFT/XC consumers: AO CUDA lowering and
  XC instead reuse the existing neutral scalar graph/emitter where allowed.
- Remove compatibility shims immediately after the move: downstream callers and
  hash-pinned reference exporters need an explicit migration window.

## Invariants

- Package movement does not itself introduce new scientific equations or retire a
  native fallback.
- IntegralIR/TensorIR mathematical serialization, equation hashes, schedules, and
  generated integral mathematics remain stable unless deliberately changed and
  requalified.
- Checkout and installed inventories use stable logical paths rather than absolute
  paths, timestamps, bytecode, or transient shim bytes.
- Compatibility leaves alias canonical module objects so enum/dataclass identity,
  `isinstance`, monkeypatching, and imports do not create duplicate class worlds.
- Generic compiler code contains no SCF policy.

## Evidence

At the #239 decomposition baseline, `cuda_lowering.py` was 6,813 lines
(~282 KiB) and `autotune.py` was 2,173 lines (~86 KiB). After decomposition, the
largest lowering leaf was `lowering/dispatch.py` at 1,274 lines (~53 KiB), and
the largest tuning leaf was `tuning/driver.py` at 704 lines (~31 KiB).

The decomposition and package migration preserved 16 generated files totaling
26,629,262 bytes exactly against revision `75472d0`. Single-run source-generation
measurements on the same workstation were:

| Generator | Before (s) | After package move (s) |
| --- | ---: | ---: |
| Production shell bundle | 2.727 | 2.827 |
| DF values | 0.180 | 0.187 |
| Weighted ERI | 0.154 | 0.158 |
| One-electron values | 2.141 | 2.149 |
| One-electron derivatives | 9.193 | 9.210 |

After integration with master `9a3aae0`, all 16 outputs also matched a pristine
checkout byte-for-byte (26,652,374 bytes); the size difference from the earlier
baseline was upstream precision-counter code. Thirteen native CPU suites and 38
selected scheduled RTX 5090 regressions passed, covering TensorIR
intermediates/layouts, grid/AO execution, and XC replay.

These measurements establish migration/build equivalence, not a runtime
performance claim.

## Consequences

Compiler ownership is easier to reason about and package independently, while
compatibility shims carry temporary maintenance cost. Artifact provenance becomes
more explicit: source inventories intentionally track canonical package paths, so
old artifacts must be rebuilt when provenance identities change even if generated
mathematics does not.

## Revisit when

Revisit package boundaries if native/runtime ownership changes, if compatibility
facades are the only remaining repository callers and downstream users have had a
release to migrate, or if a new backend requires a genuinely shared algebra layer
that cannot fit the current dependency directions.

## References

- Issue #239
- `docs/compiler_architecture.md`
- `python tools/check_compiler_structure.py`
- `docs/evidence_retention.md`
