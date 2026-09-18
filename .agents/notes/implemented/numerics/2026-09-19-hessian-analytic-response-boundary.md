# Decision: route analytic RHF Hessian response through the shared RHS contract

Status: implemented
Date: 2026-09-19

## Problem

PR #449 originally described A2 as complete analytic integration while
`cphf_relaxation` still read `System.derive()` outputs `h1`, `ERI1`, and `S1`.
Those matrices are coordinate finite differences in the independent A1 oracle.
The relaxation tensor also computed one atom-pair triangle and mirrored it, so
a raw `(R,S)` versus `(S,R)` indexing error could not reach the symmetry gate.

## Decision

The A2 response now consumes analytic first-order RHF matrices and the Hessian-
owned `build_rhf_nuclear_rhs` boundary. The bounded tools integration uses
PySCF analytic `make_h1` plus libcint overlap derivatives as its current matrix
provider; it never calls `System.derive()` for response inputs. The generated
#178 providers continue to own the explicit second-derivative skeleton.

For each nuclear perturbation, transform the analytic frozen-Fock and overlap
derivatives to MO space, build the known metric-density connection with
`metric_density_response_mo`, apply the RHF Coulomb/exchange map to that metric
density, and pass all three terms to `build_rhf_nuclear_rhs`.
The #179 operator solves the contract directly as `A x = -b`. In the symmetric
metric gauge, reconstruct the occupied-column MO derivative as
`U_ai = x_ia.T - S_ai/2` and `U_ij = -S_ij/2` before the relaxation contraction.
Evaluate every ordered atom pair independently. Do not mirror or symmetrize the
raw response tensor before measuring its symmetry defect.

## Rejected alternatives

Keeping the hand-assembled transformed RHS would duplicate the Hessian-owned
contract and permit finite-difference data to leak back into A2. Mirroring one
triangle would make the required raw-symmetry acceptance test tautological.
Using the semi-numerical A1 reference as the analytic provider would also erase
the intended independence between the oracle and the production response path.

## Invariants and evidence

`cphf_relaxation(System(mol))` must run without `System.derive()`; the regression
asserts that `h1`, `S1`, and `ERI1` do not exist. Reintroducing any of those
sources therefore fails immediately. The raw relaxation symmetry defect is
checked before any optional presentation operation. On the current H2, water
STO-3G, and water s+p+d fixtures it is below 1e-11 at the largest case.

The semi-numerical A1 relaxation remains a useful cross-check, but its first-
derivative finite-difference floor is now visible: water s+p+d differs by about
4e-9 at the default 1e-5 step, and shrinking the step increases cancellation
error. The gate is therefore 1e-8 while the independent raw-symmetry gate is
2e-10.

## Revisit when

Replace the bounded PySCF analytic first-order matrix provider with a native
generated VibeQC matrix provider once that adapter is available. Keep the same
RHS/gauge contract and no-FD regression so the substitution cannot change the
response semantics.

Agent: ChatGPT
Model: GPT-5.6 Sol


## Superseded provider choice

The [native first-order source decision](2026-09-19-hessian-native-first-order-sources.md)
replaces the PySCF-backed state and first-derivative provider with the existing
native RHF export and generated S/T/V/ERI first derivatives. The RHS/gauge and
raw-symmetry decisions above remain in effect. This note is retained as the
historical rationale for the intermediate integration.

Agent: ChatGPT
Model: GPT-6 Astra Pro
