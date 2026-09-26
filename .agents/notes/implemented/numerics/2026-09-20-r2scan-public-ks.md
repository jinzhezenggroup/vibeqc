# Decision: promote r²SCAN through the shared public KS path

Status: implemented
Date: 2026-09-20

## Problem

The generated r²SCAN scalar/XC slice already established the tau convention and fixed-density
generalized-KS pullback, but public RKS/UKS execution still needed to prove that the same
semilocal primitive could flow through the native CPU/CUDA KS owners without a private
meta-GGA driver. That promotion changes numerical behavior because the AO bilinear gains
a kinetic-energy-density contribution and the CUDA XC layout must retain the extra feature
state consistently across SCF, replay, and snapshots.

## Decision

Promote r²SCAN as semilocal functional id 2 through the existing RKS/UKS KS runtime. Preserve
the established tau definition

```text
tau = 1/2 sum_{mu,nu} D_{mu,nu} grad(phi_mu) dot grad(phi_nu)
```

and therefore assemble the generalized-KS weak-form contribution as

```text
(vtau / 2) grad(phi_mu) dot grad(phi_nu).
```

The factor is owned once by the shared semilocal XC coefficient/pullback contract; CPU and
CUDA consumers must not introduce a second tau normalization.

RKS uses the total restricted density under the existing restricted-spatial convention.
UKS evaluates alpha/beta tau and XC coefficients with the same spin conventions as the
generated r²SCAN primitive. Exact exchange remains outside this path and is neither selected
nor modified by r²SCAN.

CUDA execution generalizes the historical LDA/PBE boolean to an explicit semilocal functional
identity and carries rho/sigma/tau through the existing bounded XC/KS ownership. The public
KS snapshot records that resolved functional identity so changed-method or stale-state reuse
fails closed.

This decision promotes converged r²SCAN RKS/UKS energy/SCF execution. It does **not** promote
tau-dependent analytic nuclear gradients. The existing LDA/PBE stationary-gradient geometry
path continues to reject functional ids above PBE until the tau geometric derivative is
implemented and independently qualified.

## Rejected alternatives

- Add an r²SCAN-specific KS driver: this would duplicate MethodIR/semilocal ownership and make
  future tau-dependent meta-GGAs require another scientific runtime.
- Hide tau inside a local multiplicative XC potential: this is not the generalized-KS
  variational derivative and would miss the gradient-AO bilinear.
- Reuse the LDA/PBE geometry-gradient path for r²SCAN by ignoring tau response: this would
  publish an incomplete nuclear derivative.

## Invariants

- The tau convention remains exactly one half of the density-weighted AO-gradient contraction.
- The AO weak-form coefficient remains `vtau/2`; no backend may apply another factor of two.
- RKS/UKS spin semantics match the generated r²SCAN FunctionalSpec/MethodIR definition.
- CPU and CUDA use one semilocal functional identity rather than method-name scientific dispatch.
- Exact exchange/J/K provider selection remains unchanged.
- Analytic force capability stays fail-closed until tau-dependent AO/grid geometry terms pass
  independent directional and molecular finite-difference tests.

## Evidence

The public CPU/native selection tests exercise r²SCAN RKS/UKS through the same KS entry points
used by LDA/PBE. Generated scalar/XC contraction tests cover the r²SCAN expression, vtau, and
feature derivatives. Native CUDA XC tests compare r²SCAN energy and generalized-KS matrices
against the independent CPU path for RKS and UKS, multiple tile sizes, changed density, and
stale-state rejection. CUDA KS tests compare converged r²SCAN endpoints with CPU KS and
independently rebuild the retained physical Fock.

At PR #620 exact head `ebb7410f0c469f916d6752ed28225c4754ed754a`, Pre-commit, CuMetal CUDA,
CI, and Python wheels all passed. This evidence qualifies the public SCF/energy slice only;
it is not evidence for tau-dependent analytic forces or a performance promotion.

## Consequences

r²SCAN now reuses the common semilocal KS runtime and exposes the first public meta-GGA
vertical slice without a method-specific driver. The next #164 work is narrowly defined:
extend the shared stationary-gradient plan with the tau geometric derivative, then qualify
RKS/UKS CPU/CUDA forces. r²SCAN-3c and later tau-dependent methods can reuse that primitive
once it is complete.

## Revisit when

Revisit this decision if the tau convention changes, a Laplacian/current-dependent ingredient
requires a different operator form, or the stationary-gradient implementation shows that the
shared semilocal primitive cannot express the complete tau geometric derivative without
method-specific code.

## References

- #164
- #396
- #620
- `.agents/notes/implemented/numerics/2026-09-19-r2scan-meta-gga-generated-xc.md`
- `.agents/notes/implemented/numerics/2026-09-15-stationary-dft-gradient-boundary.md`

Agent: ChatGPT
Model: GPT-5.6 Sol
