# Decision: compile Libxc density screening into native point bindings

Status: implemented
Date: 2026-09-25

## Problem

The qualification-domain v2 array evaluator gained pinned total-density
screening, while the existing native `SemilocalPointProgram` wrapper still
executed the raw imported point graph unconditionally. Reusing the old
compiled-CPU receipt for v2 would therefore bind a new domain string to a binary
with different boundary behavior.

That mismatch is especially dangerous at exact vacuum: interpreted qualification
would return an exact zero point value, while the compiled wrapper could enter
singular interior algebra.

## Decision

Version the native point binding and compiled-CPU evidence together.

- map `libxc-bulk-production-candidate/v2` to native domain version 3;
- advance point bindings to
  `vibeqc.libxc-bulk-point-program-binding/v4`;
- retain the exact pinned `density_threshold` in binding identity;
- compile an outer `rho_a + rho_b < density_threshold` guard into the native
  point wrapper and return a zero-initialized `SemilocalPointValue` without
  entering imported scalar mathematics;
- preserve interior/version-1 and zero-gradient-candidate/version-2 mappings;
- advance compiled-CPU result/qualification receipts to v2;
- require the compiled smoke to cover both an ordinary positive-density interior
  point and exact vacuum;
- bind smoke case labels and the concatenated native output vectors into the
  receipt.

The scalar imported artifact remains the mathematical owner for active points.
Density screening is wrapper/domain policy and is therefore represented by the
binding identity rather than pretending the scalar source itself changed.

## Rejected alternatives

- Relabeling the existing v1 compiled binary as v2 would create false evidence.
- Folding density screening into every imported Graph would change canonical
  mathematical artifacts and duplicate work-driver policy inside source
  mathematics.
- Testing only a positive-density point cannot prove the new boundary behavior.
- Applying positive floors to active empty-spin/tau channels would hide genuine
  functional endpoint failures.

## Invariants

- Domain string, native domain integer, threshold, wrapper source and compiled
  receipt must describe the same execution.
- Exact vacuum screening happens before imported point mathematics.
- Active points execute the unchanged imported scalar program.
- Compiled-CPU evidence remains distinct from the independent production-domain
  scientific oracle.
- Older runtime domains retain their own native versions and semantics.

## Evidence

Focused tests lock native domain version 3, threshold retention, generated guard
source, compiled evidence schemas, exact binding identity and the
`interior + vacuum` smoke contract. The real non-curated GGA compiled test
executes both points when a host C++ compiler is available.

Repository CI is the executable validation authority for this stacked slice.

## Consequences

Once both production-domain v2 evidence and compiled-CPU v2 evidence pass for a
registration, the generic molecular-SCF qualifier can consume genuinely matching
mathematical/native execution facts rather than combining sibling domains.

## Revisit when

A shared native runtime owns the density screen outside generated point bindings,
or a later Libxc work-driver abstraction requires additional per-registration
screening state.

## References

- #1119
- #1120
- #1121
- #1333
- #1336

Agent: ChatGPT
Model: GPT-5.6 Sol
