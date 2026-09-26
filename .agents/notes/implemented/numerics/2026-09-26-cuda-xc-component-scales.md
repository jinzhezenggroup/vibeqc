# Decision: Carry MethodIR semilocal component scales in CUDA XC plans

Status: implemented
Date: 2026-09-26

## Problem

Hybrid MethodIR compositions need the CUDA semilocal owner to preserve independent exchange and correlation weights. Folding those weights into exact exchange, silently changing a functional family, or applying unqualified scaling to meta-GGA/response paths would change the scientific method.

## Decision

`CudaXcLayout` and `CudaXcPlan` carry explicit semilocal exchange and correlation scales resolved from MethodIR. The CUDA point launcher applies those weights only to the qualified PBE value path. Exact exchange remains owned by the prepared Fock provider and is not folded into the semilocal scale.

Scaled host-unfused execution, scaled response, and scaled non-PBE semilocal families fail closed. The device-chunk shortcut is also disabled whenever either semilocal scale is non-unit so it cannot bypass the qualified scaled path.

## Rejected alternatives

- Folding the hybrid coefficient into the exact-exchange owner: this would lose the independent semilocal X/C composition encoded by MethodIR.
- Treating every semilocal family as generically scalable: the existing CUDA qualification only covers the PBE value path.
- Allowing scaled response or host-unfused execution without separate validation: neither path has an independent acceptance gate in this change.

## Invariants

- Exact exchange and semilocal component scaling remain separate owners.
- Non-finite or negative component scales are rejected.
- Scaled CUDA XC is admitted only for the qualified PBE value path until additional families are independently validated.
- Response remains unit-scale only.
- Existing unit-scale LDA, PBE, r2SCAN, and omegaB97M-V behavior is unchanged.

## Evidence

The native CUDA XC test compares scaled RKS and UKS PBE (0.75 exchange, 1.0 correlation) against the independent CPU scaled-PBE integrators and verifies that scaled r2SCAN is rejected. Exact-head CI, CuMetal CUDA, Pre-commit, and PR-overlap checks passed for PR #1344 at head `e20b404f710ef79f87d91fa24145f5572455bc04` before this documentation-only follow-up.

## Consequences

PBE0 and later hybrids can compose a qualified CUDA semilocal contribution without duplicating XC mathematics or conflating it with exact exchange. Additional functional families or response scaling require their own independent numerical qualification before admission.

## Revisit when

A new semilocal family or response path has independent CPU/Libxc reference coverage and complete CUDA endpoint validation for non-unit component scales.

## References

- PR #1344
- `src/dft/cuda_xc.hpp`
- `src/dft/cuda_xc.cpp`
- `src/dft/cuda_ks.cpp`
- `tests/native/test_dft_cuda.cu`

Agent: ChatGPT
Model: GPT-5.6 Sol
