# Decision: route standard-(T) denominator sources through the canonical Fock response

Status: implemented in #155 slice A
Date: 2026-09-20

## Problem

The merged #154 response produces checked fixed-orbital RCCSD(T) weights and
separate direct derivatives with respect to occupied and virtual orbital
energies. Feeding only the ordinary CC input-block weights into the #153
gradient chain would omit the derivative of the standard-(T) denominator.
Adding that source only at the final nuclear contraction would also miss its
orbital and metric response.

## Decision

Interpret the direct `dE_(T)/d eps_p` values as a cotangent of the diagonal of
the canonical RHF Fock matrix. Reverse-differentiate that cotangent through the
same generated raw h/g/rotation Fock primal already used by the validated CCSD
gradient. This produces the denominator contribution to raw one-/two-electron
weights, symmetric metric transport, and the occupied-virtual orbital RHS
without a second handwritten derivative equation set.

Combine, before the RHF response solve:

- the ordinary CCSD baseline response;
- direct standard-(T) input-block sources;
- the corrected-Lambda contribution;
- the direct orbital-energy denominator response.

Solve one fresh total RHF Z-vector and independently replay its residual against
the generated explicit response matrix. Publish the decomposed raw h/g/S
weights and total orbital stationarity only after all identity and residual
checks pass.

The #155 A owner is response-only: it replays raw MO weights, constructs and
qualifies the RHF response operator, and builds the independent orbital matrix,
but stops before any Z solve. It does not construct AO derivative programs,
reserve complete-gradient derivative memory, or expose a nuclear-gradient
method. The CCSD(T) layer supplies the total RHS and performs the one required
Z solve only after triples and denominator sources have been combined.

## Canonical and degeneracy boundary

The denominator semantics in this slice are explicitly canonical RHF. The raw
Fock matrix must be diagonal in the retained MO basis and its diagonal must
match the bound reference orbital energies. Occupied or virtual exact/near
same-space degeneracies below the declared gate are rejected rather than
differentiating an arbitrary eigenvector gauge. A later extension may replace
this conservative rejection with a gauge-invariant degenerate-subspace
treatment.

## Scope

This slice stops at orbital/metric response. It does not contract nuclear
integral derivatives, does not return forces, does not enable the native/public
CCSD(T) force capability, and does not qualify GPU corrected-Lambda/response
execution. Those remain #155 B/C and #154 C work.

The implementation reuses the existing small-system #153 validation owner, so
its <=12-AO conventional all-electron RHF boundary remains in force.

Agent: ChatGPT
Model: GPT-5.6 Sol
