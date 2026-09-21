# Decision: bind native C++ response through a method-neutral problem

Status: implemented
Date: 2026-09-22

## Problem

The Python response stack already separates immutable response problems,
matrix-free operators and Krylov control for HF/DFT and CC consumers. The native
C++ MP2 force path was still a separate ownership seam: it constructed an
orbital-response callback and called `solve_gmres` directly. That made the
Krylov implementation itself the public integration point for a correlated
method and encouraged future C++ response consumers to repeat the pattern.

## Decision

Add `response::LinearResponseProblem` as the native C++ method-neutral boundary.
It owns only the response-space dimension and matrix-free operator action.
RHS vectors, initial guesses, diagonal preconditioners and `GmresOptions`
remain solve inputs, so one physical problem can serve multiple perturbations.

`response/solve.hpp` adapts that problem to the existing bounded true-residual
native GMRES backend. The problem contract itself has no dependency on GMRES.
The GMRES algorithm, workspace
accounting, status semantics and numerical controls are unchanged.

The conventional RHF-MP2 force path now constructs one
`LinearResponseProblem`, plans through it, and solves through it instead of
calling `prepare_gmres`/`solve_gmres` directly.

## Rejected alternatives

- Do not add another Krylov implementation: `native_gmres` remains the native
  backend and the mature Python response stack remains unchanged.
- Do not move MP2-specific orbital equations into `src/response`; only the
  matrix-free action contract belongs there.
- Do not bundle RHS data into the problem object; perturbation RHSs are
  solve-specific and future HVP/Hessian consumers need to reuse one operator.

## Invariants

- `src/response` must not depend on HF, DFT, MP2 or CC method modules.
- A response plan must match the problem dimension before any operator action.
- Native GMRES true-residual termination and workspace admission are unchanged.
- MP2 scientific equations and relaxed-gradient assembly remain owned by
  `src/posthf`.

## Evidence

- Standalone native GMRES contract executable passes after the migration.
- CPU `vibeqc` library build compiles `src/posthf/mp2_force.cpp`.
- `vibeqc_native_gmres_tests` and `vibeqc_mp2_gradient_tests` both pass in a
  CUDA-disabled CMake build.

## Consequences

Future native C++ CPHF/CPKS, correlated response, HVP or Hessian consumers can
bind a matrix-free problem without taking a dependency on the concrete GMRES
entry point. General preconditioner and multi-RHS interfaces can be added above
this boundary without changing method equations.

## Revisit when

A common native multi-RHS/recycling or resident vector engine is promoted from
the Python tooling layer into the production C++ runtime. At that point
`solve_response` should select that backend while preserving the
`LinearResponseProblem` ownership boundary.

## References

- `src/response/linear_problem.hpp`
- `src/response/solve.hpp`
- `src/response/native_gmres.hpp`
- `src/posthf/mp2_force.cpp`
- `tools/vibeqc_response/problem.py`
- `tools/vibeqc_response/krylov.py`

Agent: ChatGPT
Model: GPT-5.6 Sol
