# Decision: Bind native RKS CPKS to the live SCF state and point model

Status: implemented
Date: 2026-09-20

## Problem

Issue #179 already supplied the CPKS operator, semilocal feature contractions
and bounded GMRES/recycling. Its synthetic KS references did not prove an
actual molecular handoff. The native #162 endpoint now supplies a verified
snapshot, but its scaled SCF domain includes ordinary molecular tail points
outside the compiler's `interior-v1` response domain. Connecting identities
alone would either fail on those points or differentiate a different model.

## Decision

`NativeRKSResponse.from_native` consumes `StationaryKsState.from_native` and
its exact native owner token. It exports the existing physical orbitals/Fock
and energy, builds a matching direct CPU integral source, and uses the existing
`CPKSResponseOperator`, layout and `solve`/`solve_many` implementation.

The common XC kernel keeps tile iteration and AO assembly. Its native adapter
supplies the analytic directional derivative of the SCF Cartesian potential.
The correlation expression is shared through scalar-type parameterization;
a directional jet differentiates its existing first-derivative jet. Exchange
uses the same potential formulas for value and directional evaluation. No
additional scientific XC expression or solver is introduced. Equal-spin RKS
avoids promoting the unresolved unrestricted zero-spin second derivative.

## Rejected alternatives

- Relabeling an HF snapshot would lose native model/provider/generation proof.
- Using the interior Hessian or dropping low-density points would alter the
  discrete SCF energy model. A tiny energy error is not response acceptance.
- Finite-differencing production potentials would introduce a step-dependent
  nonlinear action into Krylov. Finite differences are validation only.
- Checking only on operator actions would allow stale zero-RHS and blocked
  zero-RHS solves to report success. Solver entry/publication needs a lease hook.

## Invariants

The actual native token, physical residual, functional, grid, packed basis,
provider and generation identities remain binding. Replayed/closed/failed
owners cannot renew an old response or recycle space. AO and batch are borrowed;
the adapter owns its integral source and snapshot lease. Exact vacuum supports
only a zero direction; unsupported directional arithmetic fails explicitly.
CPU RKS support does not qualify CUDA, UKS, ECP, DF or hybrid/meta-GGA response.

## Evidence

`tests/python/test_response_native_rks.py` covers native LDA/PBE water with
independent libcint/Libxc actions and true residuals, three finite-rotation
steps, and three reconverged one-electron perturbation steps. It also checks
transpose symmetry, sequential/blocked/recycled dependent RHSs, SCF-tail point
directions and live-owner/mismatched-source negatives. Existing independent
97-point SCF E/V fixtures protect the value path during the scalar refactor.

Additional native acceptance calls the private response ABI against 30
450-digit mixed-derivative fixtures from the original unscaled equations. A
pure relative gate fails at high-density small-gradient PBE points because
exchange and correlation cancel in the gradient response. At rho=1e10, one
component is 2.621282343339951e-15 and its FP64 error is 3.86e-23. The fixture
therefore also records independent |delta_X|+|delta_C| values. The gate adds
32 machine epsilons times this component scale to a 3e-11 relative bound and
8 minimum subnormals. This replaces an arbitrary absolute zero-gradient
tolerance and avoids weakening tiny nonzero tail checks. Production arithmetic
is unchanged; the direct ABI test also covers error boundaries, batch layout,
and energy-owner revocation from actual LDA/PBE H2 solves.

## Consequences and revisit conditions

This is a host-controlled CPU handoff. The existing solver budget excludes
operator/native-provider storage; no full endpoint performance or complete
memory bound is claimed. Extend UKS only with independently qualified spin
directions/endpoints, and device execution only with its own resident action
and resource evidence. Future method/point-model changes must update the
binding and their acceptance together.

## References

- [Issue #179](https://github.com/jinzhezenggroup/vibeqc/issues/179)
- [Response contract](../../../../docs/response.md)
- [SCF point domain](../../../../docs/xc_scf_domain.md)
