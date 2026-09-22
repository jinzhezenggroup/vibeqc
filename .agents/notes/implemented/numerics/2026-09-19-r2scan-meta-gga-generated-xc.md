# Decision: r²SCAN extends the generated semilocal XC primitive

Status: implemented
Date: 2026-09-19

## Problem

The DFT feature layer already produced tau, but the audited XC inventory was
LDA/GGA-only and treated tau derivatives as zero. r²SCAN needs nonzero tau
partials, piecewise alpha switching, and a generalized-KS weak-form tau term
without creating a method-specific KS driver or touching exact-exchange/J/K
providers.

## Decision

Represent Libxc 7.0.0 r²SCAN exchange and correlation as audited
`MGGA_X_R2SCAN` and `MGGA_C_R2SCAN` components of the existing
`SemilocalXC` primitive. The functional requires `rho/sigma/tau`; tau keeps
the existing `1/2 grad(phi) D grad(phi)` convention.

Add one shared lazy `select_le` scalar-DAG primitive. Interpreters and C/CUDA
lowering evaluate only the selected branch, and symbolic differentiation keeps
the predicate while differentiating branch values. This preserves the
r²SCAN alpha branches without smoothing or eager invalid arithmetic.

Fixed-density energy/potential uses the existing compact AO pullback, including
`(vtau/2) grad(phi_mu) dot grad(phi_nu)`. Tau-dependent response and geometry
remain fail-closed; this slice does not claim analytic nuclear forces, CPKS,
Hessians, or public molecular r²SCAN KS execution.

## Rejected alternatives

A named r²SCAN KS driver would duplicate MethodIR/semilocal ownership. Eager
`where`-style branches can evaluate singular inactive expressions. Approximating
the switching function or silently omitting vtau would change the functional.

## Invariants

- SCAN, rSCAN, and r²SCAN remain distinct definitions.
- No exact-exchange/J/K provider is selected by r²SCAN.
- New tau-dependent semilocal methods reuse the same ingredient and AO pullback.
- Unsupported tau geometric/response derivatives fail explicitly.

## Evidence

Pinned PySCF 2.14.0 / Libxc 7.0.0 fixtures validate energy, vrho, vsigma, vtau,
and every feature Hessian element. Physical alpha=0/2.5 and low-density fixtures
exercise the piecewise path. A density-matrix directional derivative detects a
missing vtau AO term. Focused scalar/MethodIR/vtau tests pass 105 cases; the
combined XC/MethodIR/KS/native suite passes 244 cases with 4 optional skips.
The CUDA compile validation tier passes source and compilation for polarized and
unpolarized order-2 split candidates on sm_120; this is compilation evidence
only, not numerical promotion or a performance claim.

## Revisit when

Promote response/geometry or public r²SCAN KS only after the corresponding
tau-dependent stationary-gradient/CPKS and backend numerical gates are complete.

## References

GitHub #164, #161, #396; Libxc 7.0.0 r²SCAN sources in
`manifests/libxc/7.0.0/`.
