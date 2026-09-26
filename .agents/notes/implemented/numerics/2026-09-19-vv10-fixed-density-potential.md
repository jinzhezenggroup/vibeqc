# VV10/rVV10 fixed-density KS/Fock potential slice

> The original rVV10 equations below used the unrevised kernel. See the
> [kernel and potential correction](2026-09-19-rvv10-kernel-correction.md).
> Original VV10 remains unchanged; old rVV10 checks are historical only.

## Scope

Issue #491 now has an executable fixed-density nonlocal-correlation potential in
addition to the slice-A energy oracle. The implementation derives vrho and
vsigma from the same discrete VV10/rVV10 double-integral definition, assembles
the total-density AO matrix, and composes it through FixedDensityMeanField for
MethodIR graphs containing NonlocalCorrelationPrimitive.

Agent: ChatGPT
Model: GPT-5.6 Sol

This is a correctness/execution slice, not the production large-grid backend.
The CPU pair loop remains quadratic in work and is guarded by an explicit
max_points admission bound. Native large-grid CPU/CUDA lowering remains #491 D.

## Derivative definition

For E = beta sum_i w_i rho_i + 1/2 sum_ij A_i A_j phi_ij, A_i=w_i rho_i,
the fixed-grid feature derivatives include both the explicit A_i factor and
the local dependence of phi through omega_i and kappa_i.

The AO contribution uses the same GGA chain-rule form for total density:

V_mn = integral [ vrho phi_m phi_n
                + 2 vsigma grad(rho) dot grad(phi_m phi_n) ] dr.
For unrestricted input the nonlocal functional consumes rho_a+rho_b, so the
same total-density derivative is published on alpha and beta AO channels.
No spin-density-only VV10 term is introduced.

## Evidence

The reference suite checks two derivative boundaries independently:

1. Direct feature perturbations of rho and grad(rho) versus analytic
   vrho/vsigma contractions.
2. Symmetric AO density-matrix perturbations versus the assembled nonlocal
   AO potential.

Both central-difference checks reach roughly machine-precision agreement on
the small fixtures. End-to-end PBE+VV10 fixed-density RKS and UKS tests then
check the complete MethodIR mean-field identity and dE = Tr(F dD), including
J plus semilocal PBE plus the nonlocal contribution.

A fresh CPU libvibeqc.so was built from this exact worktree with CUDA disabled;
no unrelated branch library was borrowed.

## Validation

- ruff check on changed Python sources: pass.
- ruff format on the changed slice: pass.
- Compiler structure: 208 modules, 0 dependency errors.
- git diff --check: pass.
- Relevant regression set with the fresh CPU library: 82 passed, 2 skipped.
- The pre-existing PBE/PBE0 fixed-density execution tests remain green.

## Capability boundary

NonlocalCorrelationPrimitive now advertises energy and ks-potential only.
It still does not advertise nuclear gradients, response/Hv, or native
production CUDA execution. FixedDensityNonlocalCorrelation fails closed above
its explicit max_points bound. Public/native KS method registration is not
expanded by this slice.
