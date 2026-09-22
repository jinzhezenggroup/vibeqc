# Decision: bind complete WB97M-V to one native KS owner

Status: implemented internal CPU energy slice
Date: 2026-09-22

## Problem

The semantic KS execution-plan boundary carries SR/LR exchange and VV10, and the production
B97M evaluator existed, but only the PBE family consumed the two-Fock route.
Accepting B97M as PBE or appending a post-SCF VV10 energy would change the
Hamiltonian without changing its advertised MethodIR.

## Decision

Add internal semilocal family 4 and point-domain version 3. Reuse the existing
RKS/UKS iteration, physical-Fock and finalization paths. Each density evaluation
builds full-range J + 0.15 K, a separate 0.85 long-range K correction at omega
0.3, B97M semilocal E/vxc/vtau, and self-consistent VV10 E/vxc. These physical
exchange fractions are converted by the common restricted/unrestricted Fock
conventions, not applied a second time by the new callbacks.

Generate the canonical SR/LR/omega and VV10 constants from the same MethodIR
that generates B97M. Compare resolved, canonical Fock requests (not raw absent
terms) and reject missing, scaled or parameter-mismatched components before
execution. The public method remains reserved; no analytic force or CUDA
endpoint is promoted by this internal composition.

## Explicit nonlocal density domain

The first full molecular grid exposed zeros and underflow in the strict positive
VV10 pair domain. The WB97M-V endpoint now selects `vv10-molecular-rho-ge-1e-8-v1`,
the same inner/outer density selection used by PySCF 2.14.0 `_vv10nlc`. This is a
numerical quadrature policy, not the exact unscreened VV10 limit. The compiler
owns the threshold; the KS payload and final-state identity record the policy.
The underlying fixed-grid API and existing PBE nonlocal calls stay strict.

The integration bridge preserves the prepared extent by giving inactive points
zero weight and finite dummy features. Active density/gradient values are not
floored. Zero weights eliminate inactive points from both pair sums and the beta
term, and the AO contraction uses those same zero weights. This deliberately
retains the original pair-work bound rather than claiming compaction speedups.
Independent compact-active-set E/V parity and invalid-input tests guard the
padding semantics. New storage is included in the integration capacity report.
No derivatives across changing active-set thresholds or complete forces are
qualified by this energy slice.

## State provenance

Retain the correction strategy and nonlocal parameters in the KS model identity.
The final-state token covers the complete physical operator, not only the
primary full-range Fock. Replays revoke old tokens and keep caller descriptor
storage detached. Family-4 final-state validation requires both auxiliary models
and the generated canonical composition.

## Validation and reproduction

Build and run `vibeqc_wb97mv_scf_tests`. An optional JSONL argument exports the
exact input basis, quadrature, converged density and complete physical Fock.
`python tools/verify_wb97mv_scf.py <dump.jsonl>` checks those against independent
PySCF/Libxc 7.0.0 calculations, including a separate core-guess SCF convergence.
The compact molecular grids are matched-grid integration tests, not a claim of
basis/grid convergence or broad chemical qualification.

The native tests also exercise both spin conventions, warm replay, changed
parameter rejection, stale-state rejection and unconverged-state refusal.
PySCF is a qualification-only dependency and is never loaded in production.

Five molecular/spin endpoints agreed with independent PySCF core-guess SCF:
maximum energy-at-native-density error 3.1e-14 Hartree; maximum complete Fock
error 4.6e-14; maximum density-matrix difference 8.6e-12. CTest retains the
independent converged energies. These are small s-basis, coarse-grid fixtures;
they do not establish broad chemical/grid convergence or CUDA performance.

Refs #935 #491 #167

Agent: ChatGPT (WB97M-V integration)
Model: GPT-6 Astra Pro
