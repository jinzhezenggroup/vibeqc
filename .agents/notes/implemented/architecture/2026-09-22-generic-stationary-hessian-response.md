# Decision: generic closed-shell Hessian response boundary

Status: implemented
Date: 2026-09-22

## Problem

The shared response layer already owned both RHF CPHF and semilocal RKS CPKS,
but the Hessian nuclear-perturbation consumer required `RHFResponseOperator`
and locally reconstructed the induced Fock map as `J - K/2`. That duplicated
method physics below the response abstraction and blocked direct reuse by DFT.

## Decision

The closed-shell response operator now owns
`induced_fock(delta_density)`. The common action and Hessian perturbation path
consume that contract. RHF therefore supplies `J - K/2` through its operator,
while CPKS supplies `J + delta V_xc` through the same boundary.

Nuclear RHS algebra, metric-density response, occupied-orbital reconstruction,
energy-weighted-density reconstruction, and single/multi-RHS solves are exposed
through method-neutral `solve_stationary_nuclear_perturbation[s]` consumers.

Existing `solve_rhf_nuclear_perturbation[s]` entry points remain strict RHF
compatibility wrappers, so current RHF HVP/CUDA callers do not silently widen
their scientific domain.

## Non-goals

This change does not claim a molecular DFT Hessian/HVP endpoint. LDA/GGA still
require complete AO/grid/partition geometric directional sources and independent
#180 validation. Hybrid, meta-GGA, RSH, DF, ECP, and unrestricted second-order
paths remain separately capability-gated.

## Validation

- Synthetic CPKS nuclear perturbation proves the XC metric-density Fock response
  reaches the shared Hessian RHS and reconstruction path.
- A synthetic RHF check pins `induced_fock` to the existing `J - K/2` contract.
- Existing RHF names reject CPKS operators rather than silently widening scope.
- Focused response/HVP-plan tests and Ruff pass without a native library.

Agent: ChatGPT
Model: GPT-5.6 Sol
