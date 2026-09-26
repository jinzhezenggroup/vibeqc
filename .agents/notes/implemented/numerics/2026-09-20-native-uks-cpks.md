# Decision: Preserve spin response and empty-spin tangents in native UKS CPKS

Status: implemented
Date: 2026-09-20

## Problem

The RKS handoff in #649 qualifies equal-spin directions only. Native UKS has
independently canonical spin frames, total-density Coulomb coupling and
cross-spin correlation. Averaging its point potentials or relabeling a UHF
reference would lose the actual SCF functional and response Jacobian.

## Decision

Reuse the existing spin reference, rotation layout and orbital-action code.
The reference now permits an explicit UKS algorithm with mandatory functional
and grid identities; its historical UHF name and UHF identity remain compatible.
The UHF operator rejects UKS references. A UKS operator substitutes only the AO
Fock-response seam: J[delta_D_alpha + delta_D_beta] plus both-spin XC response.
The same GMRES and multi-RHS/recycling implementation handles the packed vector.

RKS and UKS native adapters share the state handoff and live lease. The common
XC tile/assembly kernel supplies a spin-direction bridge through the existing
scaled correlation expression and exchange potential formulas. Scales stay
fixed through both differentiation passes. No second XC expression, density
clipping, SCF rerun or canonicalization is introduced in production.

## Empty-spin and underflow policy

An empty occupied spin has no orbital rotations, so its density and gradient
directions are exactly zero. The normal exchange density Hessian is singular
and is rejected, while the tangent action and both correlation outputs remain
qualified. The PBE spin interpolation keeps its existing C2 model extension;
this does not establish a finite arbitrary unrestricted endpoint Hessian.

At rho around 1e-280 with zero gradient, rho^(4/3) underflows. Directly dividing
zero by this product rejected a finite nonzero density response. Factoring the
reduced gradient as (gradient/rho)/cbrt(rho) only when that product is zero
retains finite directional results. Ordinary E/V branch arithmetic remains
unchanged. Tests include density-only and gradient directions at that point.

## Rejected alternatives

- A second spin response solver or copied orbital action would duplicate #225.
- Restricted averaging would remove cross-spin blocks and the independent frames.
- A finite full empty-spin Hessian would silently extend the exchange model.
- A blanket absolute tail tolerance would conceal incorrectly zero responses.

## Evidence

The native test calls the actual private ABI against 48 mixed derivatives of
the original unscaled PW92/PBE energy at 450 digits. It covers density tails,
empty and near-empty spins, C2 and numerical branch connections, spin swaps,
batch layout, and undefined direction rejection. Independent X/C component
magnitudes bound cancellation roundoff without an ordinary absolute floor.

Molecular tests use actual LiH+ and H2+ LDA/PBE states with independent
libcint/Libxc actions, isolated spin inputs, finite rotations, reconverged spin
densities, independent true residuals, transpose identities, multi-RHS/recycling
and owner revocation. Existing RKS/UHF/shared-solver tests protect the reuse.

## Consequences and revisit conditions

This qualifies the all-electron direct CPU LDA/PBE UKS handoff. CUDA CPKS,
spin-resolved device J/K, ECP/DF/hybrid/meta-GGA response and complete endpoint
performance/resource qualification still require their own evidence under
#179. A future normal empty-spin response requires an explicitly chosen model,
not an inference from the present tangent qualification.

## References

- [Issue #179](https://github.com/jinzhezenggroup/vibeqc/issues/179)
- [RKS binding](2026-09-20-native-rks-cpks.md)
- [Response contract](../../../../docs/developer/response.md)
- [SCF point domain](../../../../docs/developer/xc_scf_domain.md)
