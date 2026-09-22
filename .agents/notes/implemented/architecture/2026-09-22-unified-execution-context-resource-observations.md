# Unified execution context and resource observations

## Problem

HF, KS-DFT, RCCSD, and RCCSD(T) prepared execution each read backend/device selection
from `core::ContextState` independently. Resource reporting was similarly method-specific:
KS, D4, VV10, and coupled-cluster owners exposed useful byte counts, but there was no
method-neutral execution contract that could carry backend/device identity and resource
high-water observations across methods.

This made future runtime scheduling, common workspace ownership, and compiler-driven
execution depend on method-specific plumbing.

## Decision

Introduce `runtime::ExecutionContext` as a deliberately small, copyable prepared-execution
view. It owns only the selected backend, device id, and method-neutral resource observations.
Prepared HF, DFT, RCCSD, and RCCSD(T) owners consume that view instead of reading raw
backend/device fields throughout their implementation.

Introduce `ExecutionResourceSnapshot` with separate host/device observations for numeric
capacity and explicit scratch workspace. Observations are high-water marks and are not
additive categories. A zero value means that the owner has not supplied an observation; it
must not be interpreted as proof that the physical resource cost is zero.

The first adapters report only measurements with an existing trustworthy source:

- CUDA KS records its explicit state/XC/provider device numeric ownership.
- VV10 records its explicit host/device workspace fields.
- D4 records only its explicit workspace field, never its aggregate device allocation as
  retained numeric state because that aggregate already contains workspace.
- RCCSD records the existing CPU numeric-capacity or CUDA owned-device observation.
- RCCSD(T) additionally records the explicit triples workspace.
- HF adopts the common execution identity now but reports no synthetic resource number until
  an exact method-level observation is available.

The common `PreparedCalculation` and `PreparedBatch` interfaces expose the internal
snapshot. This change does not add or change the public C ABI.

## Rejected alternatives

- **Create a new common allocator/workspace pool immediately.** The runtime already has
  bounded-workspace, allocation-counter, and resource-ledger primitives. Replacing ownership
  while changing method adapters would mix two migrations and make numerical regressions
  harder to localize.
- **Treat every existing capacity number as workspace.** Several existing fields include
  retained state, tables, or other allocations. Relabelling those bytes would make a unified
  interface semantically wrong.
- **Put execution telemetry directly in `core::ContextState`.** That would keep method
  execution coupled to API/device-discovery mutable state and invert the desired runtime
  ownership boundary.

## Invariants

- Backend selection remains explicit; this change authorizes no CPU fallback from a requested
  CUDA execution.
- Numeric-capacity and workspace observations keep their source semantics and are never
  fabricated to make all methods nonzero.
- Resource snapshot fields are high-water observations, not values to sum into a total.
- Scientific equations, convergence criteria, integral kernels, XC kernels, and CC tensor
  equations are unchanged.
- Runtime infrastructure must not depend on HF, DFT, or CC namespaces.
- Existing method-specific allocators and CUDA graph/stream ownership remain in place in this
  first slice.

## Evidence

- Full CPU `libvibeqc.so` build succeeds with CUDA disabled.
- `vibeqc_runtime_workspace_tests` passes unchanged, preserving the standalone header-only
  workspace contract.
- New `vibeqc_execution_context_tests` verifies backend/device identity, independent
  host/device numeric and workspace high-water behavior, and reset semantics.
- Existing native UHF, DFT, and DFT API regressions pass: 3/3.
- Selected Python HF/DFT/RCCSD/RCCSD(T) regressions pass: 43 passed, 71 skipped.
- CUDA 12.9.1 / sm_120 build validation is part of this change's final qualification.

## Consequences

The method layer now has a common execution identity/resource seam without forcing a CUDA
kernel rewrite. DFT and coupled-cluster owners can publish existing resource evidence through
one interface, while methods lacking trustworthy observations remain explicitly unreported.

Batch owners may still retain `core::ContextState` at rebuild boundaries where they invoke
the existing factory; that dependency is intentionally narrower than the prepared execution
itself.

This creates a stable point for later stream policy, allocator ownership, public diagnostics,
and compiler-generated execution plans.

## Revisit when

- common stream or CUDA-graph scheduling is moved into runtime;
- a shared allocator/workspace owner replaces method-local allocation;
- exact HF workspace/numeric high-water telemetry becomes available;
- execution resource observations are promoted to the public API;
- batch rebuild factories accept `ExecutionContext` directly.

## References

- Branch: `chatgpt/agent-f-runtime-context`
- Agent: Agent F (ChatGPT)
- Model: GPT-5.6 Sol

## Recovery scope and review qualification (2026-09-23)

The recovered PR #1057 includes execution identity for HF and resource adapters
for RCCSD/RCCSD(T). The DFT, VV10, and D4 adapters described in the original
migration above are **not** part of this recovered patch; their observations
remain unreported through this interface. The original validation list is
historical evidence, not a claim that the recovered head reran those gates.

Review found that the new tracker test used assertions removed by Release builds.
The recovered test now uses always-on checks and exercises real prepared HF,
RCCSD, and RCCSD(T) CPU owners across replay. CC snapshots take the same owner
mutex as execution so concurrent readers cannot observe a data race. Resource
categories remain non-additive high-water observations, not an endpoint budget.
