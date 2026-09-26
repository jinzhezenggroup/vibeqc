# Decision: keep WB97M-V semilocal and nonlocal operators separate

Status: implemented compiler slice; public runtime remains unavailable
Date: 2026-09-20

## Decision

The canonical method owns semilocal B97M exchange/correlation, 3/20 short-range
and unit long-range exact exchange at omega=3/10, and VV10 with b=6 and C=1/100
as distinct primitives. The semilocal energy, feature gradient and Hessian are
derived from one scalar DAG using the pinned Libxc 7.0.0 coefficients,
modified PW92/Stoll decomposition and tau convention.

The direct erf attenuation branch is admitted only for a<1.35 on each active
spin density. Unsupported states fail explicitly; no density/attenuation clipping
or unqualified continuation is substituted. Exact vacuum retains the existing
energy-only convention, not a claim of a finite full endpoint Hessian. The
candidate is not a production-grid or complete WB97M-V SCF/force implementation.

## Identity and rejected alternatives

Keep semilocal source/manifest identity, omega-dependent exchange, and VV10
parameters distinct. Collapsing these into one functional identifier would hide
nonlocal operators and allow an incompatible execution or derivative route.
Do not promote public capability from scalar-point agreement or CUDA emission.
Keep upstream and normalized-source hashes separate in the provenance manifest.

## Independent review evidence

The fresh source-verified review environment uses PySCF 2.14.0 / Libxc 7.0.0.
The original WB97M/MethodIR/fixed-density RSH selection passed 50 tests. An
additional 64-point-per-spin randomized oracle comparison checks semilocal energy
and all feature-gradient components, plus the complete unpolarized feature
Hessian. Existing fixed polarized-Hessian goldens remain unchanged. These are
point/IR tests, not a new device numerical or complete-endpoint performance run.
The public Calculator still explicitly rejects the reserved WB97M-V method.

## Revisit when

Add independently validated attenuation/spin/kinetic boundary continuations,
production-grid admission, native RSH state binding and complete VV10 nuclear
and CUDA execution before public method promotion. Preserve one mathematical
source for every derivative and the existing independent numerical gates.

Refs #167, #491, #396; PR #720.

Agent: ChatGPT
Model: GPT-6 Astra Pro
