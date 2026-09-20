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

Agent: ChatGPT
Model: GPT-6 Astra Pro
