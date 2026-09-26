# Decision: Range-separated hybrids start at the fixed-density boundary

Status: implemented
Date: 2026-09-19

## Problem

Issue #167 needs range-separated hybrids without claiming that native RKS/UKS
SCF or complete analytic forces exist before their providers are implemented.
The existing native Fock plan deliberately rejects short- and long-range K.

CAM-B3LYP also cannot be represented as PBE plus range-separated HF. Its
semilocal side contains B88, short-range ITYH B88, VWN5, and LYP terms.

## Decision

MethodIR now represents short- and long-range exact exchange as explicit
primitives carrying an exact coefficient and an exact inverse-bohr omega.
CAM-B3LYP resolves to 0.19 SR-HF and 0.65 LR-HF at omega=0.33 bohr^-1.
The local scalar graph is pinned to Libxc 7.0 formulas and provenance:
0.35 B88 + 0.46 ITYH + 0.19 VWN5 + 0.81 LYP. CAMH-B3LYP reuses the same
scientific components with only manifest coefficients changed.

The fixed-density exchange assembler consumes provider-produced raw K matrices;
it does not build ERIs or K. Restricted total density uses Vx=-a*K/2,
unrestricted spin density uses Vx_s=-a*K_s, and both use
Ex=1/2 Tr(D Vx). Thus one resolved primitive graph controls energy and potential.

## Rejected alternatives

Routing SR/LR exchange through the current native FockPlan was rejected because
that provider explicitly reports these operators unsupported. Treating an RSH
as a global hybrid, or substituting PBE for the CAM semilocal terms, was rejected
as scientifically different mathematics.

The direct Libxc ITYH attenuation branch is currently audited only for a<1.35.
Inputs beyond that branch fail closed instead of silently using a cancellation-
prone expression; implementing Libxc's large-a smooth series belongs in a later
numerical extension, not an unreviewed approximation.
## Invariants

- omega is part of method semantic identity and is never inferred from a name.
- SR and LR are separate operator primitives and cache/capability requirements.
- Semilocal range parameters may affect ITYH scalar formulas but never hide HF.
- Fixed-density assembly must not claim SCF stationarity or molecular forces.
- Unsupported native SR/LR lowering continues to fail closed until #167B.

## Evidence

tests/python/test_rsh_fixed_density.py pins CAM-B3LYP energy and first
derivatives against PySCF 2.11.0 / Libxc CAMB3LYP eval_xc at independent
polarized and restricted feature points. It also uses the #166 independent interval-quadrature moments to
check SR+LR=full and restricted/unrestricted exchange variational coefficients.

RSH provenance is isolated in rsh-manifest.json and rsh_expressions.py so
adding #167A does not churn the existing PBE/PBE0 functional identities.
The existing MethodIR and XC suites remain their regression gates.

## Revisit when

Revisit the fixed-density boundary when a native SR/LR K provider is available,
or when the ITYH large-a expansion is added with independent Libxc references.

## References

Issues #165, #166, #167.
