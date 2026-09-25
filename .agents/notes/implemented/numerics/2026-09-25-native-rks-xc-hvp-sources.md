# Native semilocal RKS XC HVP source composition

Status: implemented
Date: 2026-09-25

## Boundary

Merged #1251 supplies one real all-electron CPU LDA/PBE RKS nuclear direction:
the generated one-electron/Coulomb/overlap first derivatives, the native SCF
XC geometry JVP, and exactly one shared CPKS solve. The remaining XC second-order
gap is not another response solve. It is the derivative of the three XC sources
already exposed by the stationary gradient:

- `xc_ao`: AO-center motion;
- `xc_grid`: owner-grid-point motion;
- `xc_weight`: Becke partition-measure motion.

## Decision

Add a provider-owned mixed RKS Cartesian-coefficient contraction and compose it
with the existing native SCF point bridges.

The external provider owns:

- point energy `e_xc`;
- Cartesian `v_rho` / `v_grad` coefficients from the exact SCF point model;
- their right-direction derivative from
  `evaluate_rks_response_points`, with the direction containing both nuclear
  geometry motion and the already solved CPKS density response.

The compiler contraction owner retains:

- AO jet first/mixed geometry JVPs;
- density-feature first/mixed reductions;
- quadrature first/mixed measure chain rule;
- source splitting.

Only first and second XC derivatives are required. No third XC derivative,
finite-difference production path, generated scalar-functional substitution, or
second CPKS solve is introduced.

`native_rks_xc_hvp_components` reuses the exact `DirectionalRKSResponse`
identity from #1251 and fails closed if it belongs to another live operator.
The result is explicitly a three-source XC HVP contribution, not a complete
molecular HVP. Generic stationary assembly still owns one-electron, Coulomb,
overlap/Pulay, nuclear, and final source reduction.

## Validation

1. The external Cartesian-coefficient mixed contraction is compared source by
   source against the existing generated LDA/PBE mixed Graph contraction.
2. The native RKS XC HVP source arrays are compared against multi-step central
   differences of reconverged native XC analytic-gradient sources.
3. The HVP path asserts zero additional response solves and identity/Becke
   branch compatibility.

Refs #180 #1251 #964 #958

Agent: ChatGPT
Model: GPT-5.6 Sol
