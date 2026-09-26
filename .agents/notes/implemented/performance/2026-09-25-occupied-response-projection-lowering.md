# Decision: shared lowering for occupied response projection

Status: implemented
Date: 2026-09-25

## Decision
Emit bounded auxiliary-panel C^T B C helpers and occupied-space metric-layout
transforms through the existing DF-HF response generator. The following native
integration reuses the existing response adjoint; no second gradient driver or
new scalar scientific CUDA kernel is added. Runtime owns buffers and validated
state. The existing metric scaling primitive owns eigenvalue arithmetic.

Both host-contract and CUDA generation dependencies include the new module so
incremental rebuilds cannot reuse stale emitted helpers. No public runtime or
GPU context is imported by this emitter.

## Evidence
Host checks exercise emitted BLAS transpose/stride contracts, ragged tails,
error propagation and dense-vs-occupied full-rank adjoints through an independent
host BLAS stand-in. This is not native CUDA or molecular endpoint qualification.

## References
#1078, #1334. Agent: ChatGPT. Model: GPT-5.6 Sol.
