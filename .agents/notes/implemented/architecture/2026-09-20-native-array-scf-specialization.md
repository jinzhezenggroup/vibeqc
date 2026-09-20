# Decision: checked native specialization of Array SCF density

Status: implemented
Date: 2026-09-20

## Decision

Build-time Array frontend tracing produces the shared density and energy-weighted
density TensorIR. This first native consumer specializes its exact validated
contraction topology; it is not a general TensorIR-to-C++ backend. Validate operand
order, coefficient storage layout, shared input identity and contraction labels
before emitting the bounded runtime-sized loops. A different graph must reject,
not receive a new graph hash attached to an unchanged old equation.

Keep the legacy FP64 multiplication and summation order, including the occupied
factor witness. Native code does not invoke Python tracing or a NumPy interpreter.
The generator itself participates in the shared CMake/Python source inventory,
so changing emitted mathematics invalidates source-bound caches and evidence.

## Rejected alternatives and evidence

Checking only names and scientific index kinds admits orbital-major coefficients
while emitting AO-major indexing. Independent negative layout tests cover density
and weighted density; a generator-byte identity test protects the inventory.
The repaired admission emits exactly the same header for the original valid DAG.
Native parity and density-factor tests retain exact legacy evaluation checks.

This does not establish Array API conformance, general native graph lowering,
GPU density ownership or a performance improvement. Such expansions need their
own mathematical, layout and endpoint qualification.

Agent: ChatGPT
Model: GPT-6 Astra Pro
