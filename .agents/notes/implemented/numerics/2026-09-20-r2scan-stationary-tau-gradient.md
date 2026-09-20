# Decision: extend the stationary semilocal gradient plan to r²SCAN tau geometry

Status: implemented
Date: 2026-09-20

## Problem

The public r²SCAN KS slice in #620 carries rho/sigma/tau through the generalized-KS
matrix, but #164 still needs a first nuclear derivative that uses the same compiler-owned
stationary decomposition as LDA/PBE. A separate r²SCAN force driver would duplicate the
seven-source #163 equations and risk inconsistent factors between SCF and geometry.

The subtle numerical boundary is tau. VibeQC defines

```text
tau_s = 1/2 sum_mn D_s,mn grad(phi_m) dot grad(phi_n)
```

and the native `R2scanPointValue::kinetic` field is already the compact AO coefficient
`vtau_s/2`. It is not the raw feature derivative `dE/dtau_s`.

## Decision

Extend `StationaryGradientPlan`'s semilocal backend capability to admit the `tau`
ingredient while retaining the existing source inventory: one-electron, Coulomb,
XC AO-center motion, XC grid-point motion, XC partition-weight motion,
overlap/Pulay, and nuclear repulsion.

The XC geometry adapter consumes native r²SCAN point coefficients as follows:

- UKS: the per-spin `kinetic = vtau_s/2` coefficient is consumed directly.
- RKS: native alpha/beta kinetic coefficients are averaged once because
  `D_alpha = D_beta = D_total/2`.
- No second factor of one half is applied after the native point evaluator.
- The generated meta-GGA AO pullback differentiates the existing compact bilinear
  `kinetic * grad(phi_mu) dot grad(phi_nu)`; second AO spatial jets supply its
  explicit geometry derivative.
- rho/gradient and partition/grid source ownership remain unchanged.

The live KS snapshot now retains functional selector 2 as r²SCAN rather than
collapsing every nonzero selector to PBE. The stationary contract classifies r²SCAN
as `mgga` and binds rho/gradient/tau from the actual current native state.

This slice deliberately enables the shared/CPU diagnostic path first. The existing
CUDA stationary-gradient consumer still emits LDA/PBE geometry kernels, so it
explicitly rejects `mgga` until a generated meta-GGA CUDA geometry lowering is
qualified. Public r²SCAN force capability remains off.

## Invariants

- `tau_s` keeps the one-half kinetic-energy-density convention used by #537/#620.
- Native `R2scanPointValue::kinetic` remains `vtau_s/2`; adapters must not divide it again.
- RKS applies exactly one spin-average to native kinetic coefficients.
- UKS keeps alpha/beta kinetic coefficients separate.
- The seven stationary source signs and integral/Pulay weights remain identical to #163.
- Exact exchange/J/K ownership is untouched.
- CUDA, public forces, tau density response, CPKS, Hessians and HVPs remain fail-closed
  unless separately qualified.

## Evidence

Focused tests extend the existing independent stationary-XC directional suite to
R2SCAN RKS and UKS. They compare native SCF-domain point coefficients plus the generated
AO/grid geometry pullback against independently displaced scalar r²SCAN energies over
multiple finite-difference steps. A separate generated-interior parity gate converts
the compact kinetic coefficient through the same native ABI boundary, catching an
extra or missing factor of two. Stationary-plan tests verify that tau adds an ingredient
without adding another source-equation stack.

Exact-head repository CI remains the execution gate for the stacked branch. CUDA
r²SCAN gradients are rejected before compilation/execution in this slice.

## Consequences

After this slice, the remaining #164 work is narrower: qualify complete converged CPU
r²SCAN gradients against an independent molecular reference, then generate/qualify the
same tau geometry on CUDA and only then promote public energy+force capability.

## References

- #164
- #163
- #396
- #620
- `.agents/notes/implemented/numerics/2026-09-19-r2scan-meta-gga-generated-xc.md`
- `.agents/notes/implemented/numerics/2026-09-20-r2scan-public-ks.md`

Agent: ChatGPT
Model: GPT-5.6 Sol
