# Decision: Forward ECP d derivatives through the stationary task owner

Status: implemented
Date: 2026-09-22

## Problem

The ECP d-shell change was developed against the retired primitive-record CUDA
owner while upstream moved the runtime to persistent AO topology and compact
task descriptors.

## Decision

Keep the upstream nine-word task ABI and resident topology.  For multi-component
ECP AOs, Python expands AO component terms into bounded tasks and carries the
component coefficient in the task charge.  The compiler emits one canonical
derivative library plus a compact ordered-to-canonical dispatch table that
restores center and axis slots before native atom reduction.  The all-electron
path retains its qualified s/p AOT inventory.

## Invariants

Compiler code remains independent of the public runtime.  Native allocation,
validation, and task execution remain owned by `stationary_gradient_cuda.cuh`.
Component expansion contributes to the primitive-work admission bound, and no
CPU derivative fallback is introduced.

## Evidence

`python tools/check_compiler_structure.py` passes.  The derivative schedule test
suite passes (17 tests).  CUDA numerical tests require an explicitly available
CUDA runtime and were not claimed on this host.

## References

Issue #171 and PR #1030.
