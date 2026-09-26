# Decision: generate native RCCSD CUDA Lambda response entry points

Status: implemented
Date: 2026-09-25

## Problem

The native RCCSD CUDA path already generates primal iteration and replay programs from TensorIR, but Lambda-response actions still lacked equivalent generated CUDA entry points. Adding a handwritten response implementation would split the scientific owner and make later CUDA residency work diverge from the existing shared TensorIR equations.

## Decision

Extend the RCCSD native CUDA generator so the shared and independent Lambda energy-VJP and residual-J^T programs are emitted from the same TensorIR as the CPU path. The generated state ABI gains a bounded response arena plus explicit adjoint seed pointers. A generated transpose primitive is supported because the Lambda programs require it.

This slice is codegen/ABI only. Public RCCSD(T) force dispatch, host GMRES orchestration, provenance and numerical acceptance remain unchanged until a later owner-integration slice is independently qualified.

## Invariants

- RCCSD/Lambda scientific equations remain owned by TensorIR, not handwritten CUDA kernels.
- The generated response arena is distinct from primal iteration/replay arenas and is sized from generated program requirements.
- Adjoint seeds are explicit state inputs; generated entry points do not synthesize hidden host values.
- The transpose lowering validates its permutation and preserves row-major source indexing.
- No public CUDA-force capability is implied by generation alone.

## Evidence

The no-site-packages generator contract exercises all four generated Lambda entry points and checks explicit response-arena/seed ownership. The original CuMetal failure on an unsupported `transpose` node was repaired by adding validated generated lowering; current exact-head CI/CuMetal are the build/runtime authority for this slice.

## Consequences

Later CUDA Lambda ownership can bind these generated actions without duplicating response mathematics. The generated state ABI is larger and must remain synchronized with runtime owners that consume these entry points.

## Revisit when

Revisit if TensorIR gains a more general permutation primitive that subsumes this transpose lowering, or when public CUDA Lambda/CCSD(T) response ownership changes the dispatch/provenance boundary.

## References

#155, #1311, PR #1340.

Agent: ChatGPT
Model: GPT-5.6 Sol
