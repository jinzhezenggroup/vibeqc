# Decision: lower MethodIR exact exchange to the common J/K provider

Status: implemented
Date: 2026-09-19
Issues: #165, #396

## Problem

#440 made full-range exact exchange a first-class MethodIR primitive and #457
made native semilocal KS consume MethodIR, but execution still stopped at the
primitive boundary. FixedDensityMeanField explicitly rejected every exchange
term, so PBE0 was representable without any common executable energy/Fock path.

A correct first hybrid slice must not add a PBE0 method-name branch or duplicate
the HF contraction. It also must preserve the different restricted and
unrestricted raw-K density conventions.

## Decision

Add a runtime FixedDensityMethodPlan compiled from MethodIR. The lowering accepts
one SemilocalXCPrimitive and an optional full-range ExactExchangePrimitive.
The semilocal primitive feeds the existing FixedDensityXC integrator. Coulomb
and exact exchange feed the existing FockBuildSpec/FockPlan boundary.
The exact-exchange primitive stores the physical fraction a_x. The provider
stores the coefficient multiplying its raw K matrix. Therefore lowering uses:

- restricted total density: cK = -a_x / 2;
- unrestricted spin densities: cK = -a_x.

PBE0 consequently requests -1/8 K for RKS and -1/4 K for UKS. The same
FockBuildSpec controls fixed-density two-electron energy and Fock assembly, so
the coefficient cannot diverge between the two observables.

The lowering never inspects MethodIR.identifier. A custom global hybrid with a
different exact-exchange fraction uses the same code path. Exact and
density-fitted provider choices remain explicit execution inputs and are recorded
in the executable plan identity.

FixedDensityMeanField.from_method binds an already prepared FockPlan only when
its complete requested FockBuildSpec matches the compiled MethodIR plan. The
result records both semantic MethodIR identity and executable-plan identity.

## Invariants

- Existing J/K provider implementations and ownership are reused unchanged.
- No PBE0-specific scientific driver, integral kernel, cache, or HF copy exists.
- Legacy direct semilocal FixedDensityMeanField construction keeps its v1 result
  identity and unit-J/absent-K contract.
- This boundary advertises fixed-density energy and Fock only.
- It does not register self-consistent PBE0, analytic geometric gradients, or
  forces. Those remain later #162/#163/#165 work.
- Unsupported primitive families fail before numerical evaluation instead of
  being silently ignored.
- Full-range exchange is the only executable exchange operator in this slice;
  range-separated exchange remains #166.

## Evidence

A fresh CPU-only Release build was produced from the implementation worktree.
The focused regression set passed:

- 7/7 new executable-exchange tests;
- 66 passed, 10 skipped across the new tests plus existing Fock, MethodIR and KS
  option suites;
- Ruff passed on the changed Python implementation and tests.

The PBE0 test independently reconstructs the result as unit Coulomb plus
0.75 PBE exchange plus full PBE correlation plus 0.25 exact exchange. It checks
both restricted and unrestricted densities, raw-K coefficient factors, total
energy, AO Fock, and a density-direction finite difference. A negative binding
test rejects a doubled restricted exchange coefficient before evaluation.

## Consequences

#396 now has an executable primitive boundary for the first nonlocal MethodIR
primitive instead of representation-only exact exchange. #165 Agent A gains the
requested fixed-density PBE0 energy/Fock vertical slice while leaving SCF and
force claims explicit for later work.
