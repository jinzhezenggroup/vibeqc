# Decision: separate public method manifest from scientific MethodIR

Status: implemented
Date: 2026-09-20

## Problem

Method identity and public capability facts were repeated in the C ABI enum, native
registry, Python ctypes constants, Calculator name table and benchmark parsers.
Meanwhile `vibeqc_compiler.method.MethodSpec` already owns DFT mathematical
composition, but repository ownership rules intentionally keep runtime/provider
policy out of the scientific compiler.

## Decision

`methods/public_methods.json` is the canonical source for stable public method
identity, family, public properties, batch expectation and native provider class.
`tools/generate_method_manifest.py` deterministically emits the C ABI method IDs,
Python identity maps and the native C++ metadata table.

Native provider function pointers remain handwritten in `src/methods/registry.cpp`.
The registry derives executable availability and batch support from those actual
function pointers, so manifest metadata alone cannot advertise an implementation
that the native runtime cannot call.

The public manifest does not replace `MethodSpec`/`MethodIR`: those remain the
canonical source of DFT scientific composition and compiler identity.

## Rejected alternatives

- Adding ABI/provider fields to `MethodSpec`: this mixes runtime policy into the
  compiler and makes mathematical identity depend on execution plumbing.
- Automatically assigning ABI IDs from catalog order: insertion/reordering could
  silently break ABI compatibility.
- Parsing JSON in the installed runtime: checked-in generated products keep startup
  and packaging independent of repository data files while CI checks staleness.

## Invariants

- ABI IDs are explicit manifest values and are never inferred from ordering.
- Public capability availability comes from an executable native provider, not
  from a method name alone.
- Generated artifacts are deterministic and checked by `--check`.
- New DFT mathematics still requires the appropriate MethodIR primitive/expression
  and independent validation; this manifest only removes registration plumbing.

## Evidence

- `tests/python/test_method_manifest_generation.py`
- native compilation of `src/methods/registry.cpp`
- existing Calculator/native capability tests

## Consequences

Adding a public method no longer requires editing the C method enum, Python
`_METHODS` table, native registry identity/capability rows, or resource-probe name
parser independently. Provider adapters and genuinely new scientific execution
remain explicit native/compiler work.

## Revisit when

A common generated method-level runtime plan can safely resolve provider function
pointers without moving runtime ownership into the compiler, or when ABI versioning
permits replacing the current explicit integer method IDs.

## References

- #396
- #349
- `python/vibeqc_compiler/method/spec.py`
- `src/methods/registry.cpp`
