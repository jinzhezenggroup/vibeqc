# Decision: Generate RSH exchange gradient weights from MethodIR

Status: implemented
Date: 2026-09-20

## Problem

Issue #167 needs analytic nuclear gradients for range-separated hybrids without
duplicating exchange coefficients between energy, Fock, and force code. Merged
#249 already supplies first nuclear derivatives of short- and long-range ERIs,
while merged #570 owns the exact exchange coefficients and omega in MethodIR.
The remaining stationary-gradient layer needs to connect those contracts without
claiming that a complete converged RSH force endpoint exists before #167B binds
a live RKS/UKS state.

## Decision

StationaryGradientPlan accepts a semilocal MethodIR plus canonical short- and
long-range exchange primitives. It adds one explicit gradient source per range
operator and generates the exchange ERI cotangent from the primitive coefficient.

For an ordered exchange quartet (i,k|j,l), the caller binds density pairs
D(i,j) and D(k,l). Restricted total-density exchange uses
-a/4 D(i,j) D(k,l). Unrestricted exchange uses
-a/2 sum_spin D_spin(i,j) D_spin(k,l), with no alpha-beta cross terms.

The range operator and fixed inverse-bohr omega remain properties of the MethodIR
primitive. Omega is part of the stationary-plan identity and is held fixed under
nuclear differentiation. The actual range ERI derivative remains owned by the
#249 weighted-ERI provider; this change does not create a second RSH J/K backend.

## Rejected alternatives

- Reusing the Coulomb source weight would incorrectly sum alpha and beta before
  multiplication and would introduce cross-spin exchange in UKS.
- Hard-coding CAM-B3LYP coefficients in the gradient path would allow energy,
  Fock, and force semantics to drift.
- Routing range derivatives through first_gradient.py was rejected because that
  legacy executor explicitly accepts full Coulomb only; #249 is the range-aware
  provider.
- Enabling a public molecular-force capability here was rejected because the
  converged native RSH SCF/state binding is still owned by #167B.

## Invariants

- Exchange coefficients come only from the resolved MethodIR primitive.
- RKS uses the occupation-two total-density convention; UKS exchange is
  same-spin only.
- Short- and long-range sources remain distinct and preserve exact omega identity.
- Nuclear derivatives hold omega fixed.
- Missing exchange sources fail the complete-source reduction gate.
- This compiler plan alone never grants a public CPU/CUDA force endpoint.

## Evidence

tests/python/test_stationary_gradient_plan.py checks both range operators,
RKS/UKS coefficients, absence of cross-spin exchange, three-step finite
differences, source coverage, and omega-sensitive identities.

The existing #249 range-separation suite continues to validate generated range
integrals and first derivatives independently.

## Consequences

The analytic exchange-force mathematics can be developed and reviewed in
parallel with native RSH SCF. Once #167B supplies a validated converged state,
its density and #249 provider can bind directly to these generated source
weights without redefining RSH force coefficients.

## Revisit when

Revisit this boundary if omega becomes a differentiated/optimized variable, or
if a screened production provider changes the exact mathematical operator rather
than only its schedule.

## References

- #167
- #249
- #570

---
Agent: ChatGPT
Model: GPT-5.6 Sol

## Integration with the full-range stationary owner

The master integration preserves the existing full-range `exact_exchange`
primitive and its `fock_coefficient(spin)/2` weighting alongside the separately
named SR/LR sources. A sole range primitive is not a global exchange primitive;
class-based lookup prevents a two-node MethodIR from being misclassified.
Non-RSH source order, payload and block identities remain exactly those of the
existing stationary owner. Only a plan with range-exchange primitives acquires
the range-specific v3 schema and fixed-omega convention.

A direct comparison against the master implementation checks 16 combinations
of LDA/PBE/r2SCAN/PBE0, RKS/UKS and all-electron/ECP envelopes, including generated
block identities. Four permanent single-SR/LR spin cases protect the distinction.
These are plan/algebra checks, not new native RSH molecular-force qualification.

Integration review: Agent ChatGPT; Model GPT-6 Astra Pro.
