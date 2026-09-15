# Scientific compiler instructions

These rules apply under `python/vibeqc_compiler/`.

## Ownership and dependency boundaries

- The compiler owns mathematical IR, lowering, schedules/planning, code emission,
  artifact identity, and finite compiler processes. Native runtime allocation and
  execution templates remain owned by the corresponding `src/` subsystems.
- Source generation must not import the public runtime, load `libvibeqc`, probe a
  GPU, or import PySCF, Torch, or CuPy. Keep generation usable from an uninstalled
  checkout with the documented compiler dependencies.
- Do not place SCF or method policy in generic compiler code.
- Reuse the existing scalar/IR algebra rather than introducing a second scientific
  algebra implementation for a new consumer.

## Determinism and compatibility

- Generated source, mathematical serialization, equation hashes, logical source
  paths, and artifact identities are scientific/build contracts. Changes must be
  deliberate and independently validated; package moves alone should not alter
  generated mathematics.
- Keep checkout and installed compiler inventories logically identical. Do not add
  absolute installation paths, timestamps, bytecode, or transient compatibility
  shim bytes to source identities.
- Compatibility facades must forward to canonical module objects rather than load
  duplicate implementations. Retire them only after downstream callers migrate
  and compatibility tests are the remaining repository users.

## Validation and rationale

Run the relevant compiler structure/generation checks for ownership or packaging
changes, including `python tools/check_compiler_structure.py` when applicable.
For non-trivial ownership, package, identity, or compatibility decisions, add an
Agent Note under `.agents/notes/` with the preserved outputs/evidence and rejected
alternatives. The package-migration precedent is recorded in
`.agents/notes/implemented/architecture/2026-09-15-compiler-package-ownership.md`.
