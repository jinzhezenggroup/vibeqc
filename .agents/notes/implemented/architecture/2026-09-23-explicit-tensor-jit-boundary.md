# Decision: defer TensorIR JIT discovery until bounded lowering succeeds

Status: implemented
Date: 2026-09-23

## Problem

The public CPU JIT wrapper constructed a compiler adapter before the canonical
lowering checked primitive support, FP64 semantics and resource budgets. For an
unsupported FP32 program, a missing compiler therefore masked the semantic
error. Writable public identity fields could also relabel provenance without
changing the native owner or its execution.

## Decision

The native CPU owner accepts either an existing adapter or a zero-argument
adapter factory. It invokes a factory only after its existing bounded lowering
succeeds. The public extension supplies that factory, keeping one authoritative
semantic/resource check and one source emission. Existing internal callers keep
their adapter-based interface. No cache directory or source artifact is written
before semantic admission.

The public handle exposes read-only logical hash, target and mode properties;
artifact metadata and resource reports remain detached copies. Importing the
extension or rejecting a public target/mode does not import JIT machinery.
Explicit compilation may import lowering to validate a program, but does not
discover or execute a compiler for rejected semantics or budgets.

## Evidence and boundaries

Tests forbid executable discovery and subprocesses for unsupported dtype,
primitive and byte/work/node budgets. A fresh subprocess checks import and
target/mode rejection laziness. Actual C++ compilation checks execution, cache
reuse, artifact hashes and hostile feeds; the existing native TensorIR tests
retain independent NumPy algebra and ABI coverage. Provenance assignment and
nested report mutation are checked separately.

This is an explicit advanced-user CPU FP64 JIT interface. Built-in methods keep
their AOT contract; CUDA and AOT extension compilation remain rejected.

## Rejected alternatives

Running emission twice in the wrapper would duplicate work. Reimplementing its
semantic checks would drift from the compiler. Globally deferring discovery in
all compiler adapters would change unrelated owners' activation contracts.
