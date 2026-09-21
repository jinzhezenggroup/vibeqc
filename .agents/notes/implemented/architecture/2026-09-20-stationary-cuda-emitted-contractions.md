# Decision: compiler-owned stationary-gradient CUDA contractions

Status: implemented
Date: 2026-09-20

## Boundary

Method-level lowering emits primitive weighting/reduction, center validation,
XC geometry contraction and final geometry reduction. The native header keeps
forward declarations, state, admission, allocations, transfers, stream launches,
metrics and ABI. Integral recurrence, AO pullback, scalar XC and Becke partials
retain their existing owners. This slice depends on the shared #626 adjoint emitter.

## Evidence and invariants

All five moved definitions are token-identical to the original native bodies.
The generator inserts definitions after the runtime include; declarations resolve
the launch sites without retaining a second scientific definition. Preserve
component weights/signs, all source contributions, deterministic reduction,
finite-value admission and failure-atomic output. Source/strict-mode tests must
continue rejecting implicit runtime imports and NVCC arithmetic overrides.

## Alternatives and limits

Duplicating the kernels or changing reductions during relocation was rejected.
The changed source closure deliberately changes generated artifact identity.
This does not expand method capability or prove a faster/full-resident endpoint.
Revisit only with independent numerical and complete-endpoint qualification.
Refs #349/#163/#626; tests/python/test_stationary_cuda_lowering.py.

Follow-up: the host-visible generated source-weight handoff was superseded by
`.agents/notes/implemented/performance/2026-09-20-stationary-weight-consumer-fusion.md`
under #665; the compiler-owned scientific-contraction boundary remains unchanged.

Agent: ChatGPT
Model: GPT-6 Astra Pro

## Native task-batch integration (2026-09-21)

The #664 task-batching slice keeps the #665 resident D/W source-weight boundary.
Python submits bounded AO tuple descriptors plus only the physical nuclear charge;
the emitted CUDA task kernel evaluates the current `StationaryGradientPlan` source
weight from resident density/weighted-density before enumerating primitive products.
Immutable primitive/AO topology is uploaded once per prepared owner, while geometry
reset refreshes centers and the current D/W frame. This avoids restoring the retired
host source-weight handoff while removing expanded primitive-record uploads.

The runtime therefore admits both spin-block identity and per-reset primitive work,
and prepared geometry replay must preserve the immutable topology identity. The
source-owner byte formula includes resident D/W, topology, compact tasks, and task
outputs; task metrics remain cumulative while the primitive-work cap is per reset.

Agent: ChatGPT
Model: GPT-5.6 Sol
