# Decision: cache stationary primitive code independently of method wrappers

Status: implemented candidate; full endpoint qualification remains separate
Date: 2026-09-21

## Problem

The large generated primitive derivative implementation is reused across XC
functional/spin wrappers. Recompiling it for every wrapper duplicates compiler
work without changing its mathematical program. A reused object must not become
an unchecked external binary or lose the target/precision/source contract.

## Decision

Compile relocatable primitive and wrapper CUDA objects under independently
hashed source/header/toolchain/option identities. Verify object bytes on reuse
and again before device linking. Publish completed objects atomically under the
existing cache lock. Primitive split-compile options belong to its compile step,
not to an accidentally retained whole-source compile call at device link.
The native artifact still owns the final linked library and its identity.

## Compatibility and evidence

The integration retains master's compact native task keys and failure-recovery
fixture. CPU allocation remains an appended optional field, preserving positional
Slurm time arguments; invalid boolean/fractional/nonfinite counts reject before
scheduler construction. Multi-node/multi-task duplicate benchmark execution is
still rejected. A three-way merge initially duplicated the CPU field; the
existing positional-compatibility regression caught that and remains enabled.

The integrated host profile/lowering/task-budget suite passes 71 cases. The full
stationary r2SCAN artifact was compiled and linked by NVCC 12.9.1. A separate
actual allocated-GPU two-object test reuses one unchanged primitive object for
two wrappers, executes both expected results, then rejects a corrupted object.
These are not complete molecular force or endpoint speedup measurements.

## Revisit when

Promote only after the exact integrated molecular force/replay and cold/warm
compile/endpoint/resource gates are complete. Do not make cache hits bypass byte
verification or use a different precision/target policy merely to reduce time.

Agent: ChatGPT
Model: GPT-6 Astra Pro
