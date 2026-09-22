# Prepared CUDA Fock execution seam

Agent: ChatGPT (Agent A — Provider / DFT integration)
Model: GPT-5.6 Sol
Date: 2026-09-22

## Decision

CUDA method consumers must not borrow a concrete Direct-J/K handle merely to
execute the Fock model already owned by `PreparedFockPlan`. A new narrow
`PreparedCudaFockBinding` plus `enqueue_prepared_cuda_fock` seam exposes
device/stream/source identity and device-resident raw J/K execution while
leaving provider selection, scientific coefficients, screening, ownership and
resource admission with the prepared Fock owner.

## First consumer

`dft::CudaKsPlan` now uses only this prepared execution seam for its
ordinary-stream J/K binding and enqueue. It no longer includes the Direct-J/K
device header, stores `CudaDirectJkPlan*`, or calls Direct-J/K stream/device
helpers. The solver-region replay identity receives the provider's opaque
owner-local source identity through the same binding.

This slice deliberately preserves the existing CUDA KS scientific admission:
only the qualified conventional Coulomb-only semilocal route is accepted by
`CudaKsPlan`. The seam itself also admits an already-prepared full-range exact
J/K owner so a later hybrid slice can consume K without another DFT-specific
Direct-J/K dependency.

## Boundary and follow-up

The first implementation is the existing full-range exact CUDA provider.
Unsupported DF/mixed-provider compositions fail closed; no fallback or provider
substitution is added. A later #930 slice may teach the same seam additional
resident providers. CUDA PBE0/B3LYP promotion remains separately gated on
semilocal scaling, K/Fock/energy assembly, convergence/final-state validation,
and endpoint evidence; this refactor does not advertise that capability.
