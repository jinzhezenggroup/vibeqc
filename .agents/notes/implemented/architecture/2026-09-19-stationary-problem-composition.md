# Decision: explicit stationary equations before native adjoint composition

Status: implemented
Date: 2026-09-19

## Problem

#181 needs a reusable objective/residual/constraint/source graph above TensorIR.
A local derivative generator cannot infer the SCF orthogonality constraints, CC
residual definition or upstream provider dependencies from an energy expression.
The ongoing #465 primitive already owns implicit-response execution, so another
solver or method-specific derivative stack would duplicate work.

## Decision

Add `method.stationary` as a compiler-only composition boundary. A problem binds
independent state inputs to physical residual outputs, declares constraint blocks
and field-level provider identities, and generates the explicit Lagrangian,
negative objective-state gradient and Lagrangian source partials through #151.
Provider ancestors are a validated dependency/pullback inventory, not executable
custom rules. Generated weights stop at TensorIR input fields. The schema exposes
what still needs a provider pullback rather than calling partials molecular forces.

The schema is based on master and does not depend on unmerged #495. A later #465
adapter can consume its state/equation/source contracts without a second solver.
The existing #163 stationary mean-field plan and #466 spectral rule are unchanged.

## Rejected alternatives

- Differentiate the SCF/DIIS iteration history: this confuses equation and solver
  semantics and retains a history-sized tape.
- Infer constraints or a physical state from E alone: this silently drops terms.
- Declare a provider rule name and treat its derivative as executed: source
  cotangents are not nuclear gradients, and direct plus propagated contributions
  must later be accumulated once at each upstream field.
- Flatten packed/redundant state coordinates with an unweighted dot product:
  this loses orbit metrics and can produce an incorrect transpose solve.
- Implement another response solver or a large ProgramIR: #465/#179 and #460 own
  those different boundaries.

## Invariants

Use `L=E+lambda^T R`, `R_x^T lambda=-E_x`, and `E_q+R_q^T lambda` consistently.
KKT constraint multipliers are explicit state blocks. A state absent from the
objective needs an exact zero RHS; #151 must see the original multi-output primal
with its residuals, since dead retained definitions are not live AD inputs.
No state Jacobian or global nuclear-coordinate Jacobian is constructed.
Unknown provider edges, cycles, duplicate physical fields and omitted declared
sources fail explicitly. Data-only replay regenerates derivative fragments.
Model/gauge/layout/provider/solver-contract changes invalidate the problem identity.
No new public capability, native/GPU endpoint, runtime dependency or release is added.

## Evidence

`tests/python/test_stationary_problem.py` independently checks a re-solved
nonvariational scalar problem, a nonsymmetric linear system with matrix and vector
parameters, and a constrained quadratic/KKT system. Their derivatives are compared
with separately written analytic formulas and re-solved finite differences.
Other regressions cover source graphs, corrupt replay, frozen records, repeated
input aliases, dense symmetric source projection, failure domains, and a
2000-coordinate diagonal problem with no dense Jacobian. CUDA checks exercise
planning only. The associated PR records actual test counts for its pinned head.

## Consequences

The first slice is directly testable before #465's full native integration but
cannot certify convergence, state freshness, invertibility, total provider-chain
response or joint runtime memory. Its generated stationarity residual is an
adjoint diagnostic only. Method-specific oracles remain independent.

## Revisit when

A qualified #465 adapter binds a native HF/RI-MP2 state and solver to this schema;
provider VJP dispatch and coordinate contractions can then be composed and checked
against complete molecular finite differences. Non-Euclidean/packed coordinates
and higher derivative orders need separately qualified adapters, not silent flags.

## References

#181, #151, #163, #179, #193, #396, #460, #465, #466, #495.
`docs/stationary_problem.md` describes the current compiler contract.

Agent: ChatGPT
Model: GPT-6 Astra Pro
