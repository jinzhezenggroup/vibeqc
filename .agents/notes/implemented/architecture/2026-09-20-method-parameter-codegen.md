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
