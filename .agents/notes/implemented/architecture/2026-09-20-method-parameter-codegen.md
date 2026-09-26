# Audited method parameter codegen

Agent: ChatGPT
Model: GPT-5.6 Sol

## Decision

Method-level scientific constants now have one editable source:
`python/vibeqc_compiler/method/method_parameters.json`. The source records
parameters together with pinned provenance/data identities. Product code does
not parse this JSON at runtime.

`tools/generate_method_parameters.py` validates the source and emits two
compile-time/runtime-free views:

- the committed Python `_generated_parameters.py` used by MethodIR factories;
- a C++ `generated_method_parameters.hpp` produced by CMake for native/CUDA
  compilation.

The generated C++ accessors are `constexpr`, and CUDA-visible accessors are
host/device functions. D3(BJ), D4 and r2SCAN-3c gCP method parameters therefore
share the same source without introducing file I/O into calculation paths.

A freshness test regenerates the Python view byte-for-byte and records the
source SHA-256 in both generated forms. Parameter changes must update the JSON
source and regenerate the Python module; hand edits to generated output fail
qualification.

## Typed generated boundary

The generated catalog retains immutable raw mappings for provenance/compatibility.
Generated typed accessors return detached TypedDict records after checking every
float, string, boolean and element-tuple field. JSON generation also rejects
unknown/missing fields and invalid field types against explicit schemas.
This replaces Any casts at D3/D4/gCP constructors without importing the public
runtime, repeating scientific constants or changing the generated C++ values.
Malformed-source regressions fail on the prior generator and pass on this one.
Source identity still covers the editable JSON and generator through the shared
CMake/Python inventory.

Agent: ChatGPT
Model: GPT-6 Astra Pro

## Superseded editable-source layer

The [pinned-catalog source decision](2026-09-20-pinned-dispersion-catalog-sources.md)
supersedes the JSON-editing instructions above after #676. The JSON is now a
generated intermediate from verified upstream snapshots and explicit local
overrides. The typed Python/C++ lowering and runtime-free ownership described
here remain unchanged; this historical decision is retained rather than rewritten.

Agent: ChatGPT
Model: GPT-6 Astra Pro
