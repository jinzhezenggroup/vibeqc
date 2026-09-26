# Decision: Keep generated range-exchange stationary derivatives backend-explicit

Status: implemented
Date: 2026-09-26

## Problem
The generated short- and long-range weighted-ERI first-derivative provider was bound only to the CPU C++ compiler adapter even though the compiler/runtime already support the same bounded generated program on CUDA. Extending this path must not introduce a second handwritten derivative implementation, silently switch backends, or make a complete WB97M-V CUDA force capability claim before the downstream stationary-gradient composition is qualified.

## Decision
`RangeExchangeExecutor` accepts only an explicit `CppCompilerAdapter` or `CudaCompilerAdapter`. The adapter determines the compiled backend identity, and preparation verifies that the returned artifact reports the same backend. CUDA execution also receives an explicit non-negative int32 device ordinal; CPU execution requires device 0.

The existing generated weighted-ERI first-derivative mathematics, component subset, bounded record/tile runtime, and resource planner remain the scientific owner. This change exposes only the SR/LR stationary derivative provider seam. It does not promote a complete CUDA range-separated-hybrid or WB97M-V force endpoint.

## Rejected alternatives
- A separate handwritten CUDA RSH derivative kernel was rejected because it would duplicate derivative mathematics and create a second numerical owner.
- Inferring the backend from environment or device availability was rejected because it would make execution identity implicit.
- Treating successful compilation as complete WB97M-V force qualification was rejected because the downstream semilocal, exact-exchange, VV10, and stationary composition remain separate acceptance boundaries.

## Invariants
- Generated weighted-ERI mathematics is shared between CPU and CUDA.
- Compiled artifact backend identity must match the explicitly selected compiler adapter.
- CPU execution never accepts a nonzero CUDA device ordinal.
- The current public stationary binding remains limited to s/p public AOs.
- No complete CUDA RSH/WB97M-V force capability may be advertised from this provider seam alone.

## Evidence
At reviewed head `b0d34abcb0b5b2fd95844e829838f7ce6f7f2ae1`, the PR's CI, CuMetal CUDA, Pre-commit, and PR-overlap workflows all passed. The test suite contains an opt-in Slurm/NVIDIA comparison of short- and long-range generated CUDA derivatives against the same generated CPU provider with explicit force tolerances; normal CI does not substitute for executing that real-device gate.

## Consequences
The provider can be composed by later CUDA stationary-gradient slices without changing its mathematical owner. Callers must supply the backend and, for CUDA, the device ordinal explicitly.

## Revisit when
Revisit this boundary if the stationary consumer adds d/f public AO support, if the weighted-ERI runtime changes backend identity or resource ownership, or when a complete public CUDA RSH/WB97M-V force endpoint is qualified.

## References
- PR #1388
- Issues #167 and #1332
- `python/vibeqc/_stationary_rsh_cpu.py`
- `python/vibeqc_compiler/integral/weighted_eri_execute.py`
