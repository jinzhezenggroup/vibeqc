# Decision: generate stationary source weights before native DFT force assembly

Status: implemented (compiler/diagnostic slice only)
Date: 2026-09-19

## Problem

Issue #163 already has stationary XC partials and generated moving-grid response.
A new handwritten PBE assembler would duplicate method coefficients and spin/sign
rules between CPU and CUDA. MethodIR alone describes XC/exchange composition,
not the shared Hamiltonian and overlap constraint.

## Decision

Use the existing method package and TensorIR. An explicit StationaryMeanField
envelope supplies the all-electron/direct/fixed-occupation stationary context.
One-electron, Coulomb and overlap objectives generate integral cotangents through
TensorIR VJP. Specialize its scalar objective seed to exact one, remove dead
primal integral inputs and contract only bounded ordered-element derivative
blocks. The Pulay minus sign and Coulomb half factor originate in the objective.
The seven-source reduction rejects incomplete or duplicate component coverage.

The same TensorIR graphs have CPU-interpreter execution and existing CUDA source
lowering. No separate algebra interpreter, compiler dependency exception, public
force registration or native runtime is introduced.

## Rejected alternatives

- Handwritten derivative weights: this would copy objective factors and UKS spin
  accounting into another formula owner rather than exercise shared AD.
- One global AO-rank-four cotangent/Jacobian: providers should stream ordered
  tuples and coordinate blocks, with explicit center scatter outside this slice.
- A metadata-only plan: executable source contractions and independent numeric
  oracles are necessary before adding runtime adapters.
- Pretending source generation is GPU force completion: XC/grid integration,
  live state binding and complete molecular validation remain open.

## Invariants

Restricted D/W already include their occupations. UKS Coulomb uses the total
alpha/beta density, including cross-spin terms. The overlap source uses current
occupation-weighted W. Functional XC coefficients belong inside the resolved
primitive and must not be applied again to its geometric partials. Full ordered
tuples have no hidden symmetry multiplicity. Only the integral source is
locally differentiated; there is no SCF tape or new CPKS prerequisite.

Mathematical plan, generated-block equation/shape, backend artifact and live
lease identities are distinct. Detached algebra inputs are diagnostics, not
proof of a converged molecular state. Native endpoint capability fails closed.

## Evidence

`test_stationary_gradient_plan.py` checks independent scalar-loop source weights,
all Cartesian block components, three-step displaced-source energy differences,
UKS cross-spin and RKS total-density recovery, off-diagonal ordered tuples,
partial tiles, every missing/sign-flipped reduction source, duplicate coverage,
resource/type/nonfinite rejection, semantic identities, replay and deterministic
CUDA source lowering. These are algebra/source-generation gates, not molecular
or real-device force evidence. Existing XC/grid and state gates remain separate.

## Revisit when

Bind this plan to the actual current native KS state and independently qualified
integral/XC/grid providers in #163 B2.2/C. Extend primitive rules for new physics
rather than add named-method dispatch. Preserve the independent oracles when
introducing optimized packed providers or alternative native schedules.

Refs #163, #396, #455, #473.

Agent: ChatGPT
Model: GPT-6 Astra Pro
