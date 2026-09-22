# Decision: one compiled-execution lifecycle across CUDA consumers

Status: implemented
Date: 2026-09-21

## Problem

TensorIR CUDA Graph replay (#507), prepared stationary-force reuse (#663), and
the bounded CUDA KS device-control chunk (#370) had compatible lifetime needs but
different owner-local state machines. Extending each path independently would
duplicate binding invalidation, warm/failure state, recovery, and execution
identity rules before any additional replay backend was qualified.

The KS chunk already removes a host decision between two physical iterations,
but direct-J/XC capture safety and host generation bookkeeping are not yet proven.
Treating the chunk as a CUDA Graph merely because both are retained execution
would therefore conflate control-flow residency with graph replay.

## Decision

Introduce `runtime::CompiledExecutionRegion` as the method-neutral native
lifecycle for a compiler-qualified CUDA execution region. A binding contains the
qualification identity plus owner-local device, stream, arena, and library
addresses. The lifecycle owns binding transitions, warm state, sticky failure,
explicit recovery, invalidation, and cumulative diagnostics.

`CudaGraphRegion` now composes that lifecycle instead of maintaining its own
`bound/warmed/failed` flags. Its capture/instantiate/replay resource remains a
backend detail and is still destroyed before the buffers/stream/library it
references.

The existing two-slot CUDA KS device-control path binds the same lifecycle when
it is selected. Successful chunk completion records one execution; submission,
completion, or device-control failure records a sticky region failure; a later
compatible `begin()` explicitly recovers the region before further work.
Ordinary host-controlled KS never binds this region.

At the pure compiler/runtime-identity boundary,
`CompiledExecutionIdentity` canonicalizes request, artifact, and runtime payloads.
Prepared TensorIR owners publish this identity without changing their historical
public identity. Prepared stationary CUDA force owners use the same contract as
their retained execution identity, replacing an owner-specific
`sha256(repr(...))` construction.

## Rejected alternatives

Do not capture the KS chunk in this change. Direct-J and XC enqueue paths have
host-side generation/lifetime semantics whose graph-replay legality has not been
proven. A graph launch must never silently replay stale generation state.

Do not move SCF convergence, DIIS, occupation stabilization, final-state
stationarity, or failure semantics into the generic lifecycle. Those remain
method-owned. The shared lifecycle records whether a qualified region can be
reused; it does not decide scientific legality.

## Invariants

- One failed or changed binding cannot reuse a compiled backend silently.
- Graph capture failure remains sticky until graph invalidation, preserving the
  existing no-repeat-on-launch-error rule.
- KS region recovery occurs only at a new solve boundary after method state has
  been reset by `begin()`.
- Pointer values are owner-local native binding facts and never enter portable
  compiler artifact identity.
- The portable execution identity contains no raw pointers and uses canonical
  JSON hashing.
- Ordinary KS and ordinary TensorIR execution remain valid fallbacks.
- No new method-specific graph cache or invalidation stack is introduced.

## Evidence

CPU/source validation for the shared lifecycle uses the fake CUDA runtime in
`tests/python/test_cuda_graph_region.py`; it exercises bind, repeated bind,
binding change, success, failure, recovery, invalidation, graph warmup/capture,
replay, fallback, and launch-error no-duplicate semantics.

`tests/native/test_ks_cuda.cpp` checks that opt-in two-slot RKS chunks publish
one shared-region execution per chunk while ordinary KS leaves all region
counters zero. Its existing numerical endpoint and bounded-speculation gates
remain unchanged; failure/recovery diagnostics are also checked.

Prepared stationary-budget tests and compiler-structure validation cover the
portable identity migration without constructing a GPU owner.

## Consequences

The execution lifecycle is now reusable independently of the backend used to
submit work. CUDA Graph is one backend; the KS device-control chunk is another.
A future ProgramIR/SCF/CC region can therefore share lifecycle and portable
identity before choosing graph, persistent-kernel, cooperative, or ordinary
stream execution.

This change alone does not claim a new KS speedup and does not auto-promote the
two-slot path. It removes duplicated ownership machinery and makes a later
capture-safety qualification an isolated backend decision.

## Revisit when

Direct-J/XC can prove capture-safe generation semantics, resource accounting
includes graph-retained state, and complete cold/warm/changed-geometry endpoint
measurements justify a graph-backed KS region. At that point add a replay backend
under the existing shared owner rather than another KS-specific lifecycle.

## References

#507; #663; #370; #459; #682;
`docs/ks_diagnostics.md`;
`.agents/notes/implemented/architecture/2026-09-19-compiled-cuda-replay.md`;
`.agents/notes/implemented/performance/2026-09-20-cuda-ks-iteration-chunks.md`.

Agent: ChatGPT
Model: GPT-5.6 Sol

## Review integration clarification

The integrated stationary CUDA owner retains `PreparedExecutionLease` as the
source of its public `identity`, replay admission and failure/recovery state.
`compiled_execution_identity` is additional portable owner metadata, derived
from the resolved target and basis topology; it does not replace or authorize
that lease. The earlier description of replacing the retained execution identity
predates this integration. TensorIR publishes the new identity in execution
metrics while preserving its historical public identity.
