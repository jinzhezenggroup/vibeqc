# Decision: bind corrected standard-(T) response to verified sources

Status: implemented
Date: 2026-09-20

## Decision

Standard noniterative triples change the Lambda right-hand side, not the CCSD
residual Jacobian. Solve `J_CCSD^T lambda_total = -d(E_CCSD + E_(T))/dt` through
the existing transpose solver and independently replay both generated forms.
Dense triples cotangents pass through the declared packed-coordinate adjoint.
Fixed-orbital weights combine the CCSD baseline, direct triples contribution and
`delta-Lambda * dR_CCSD/dq`. Denominator energy weights remain separate inputs to
later orbital response, not complete nuclear forces or physical RDMs.

## Binding and rejected alternatives

Recompute the source identity from the actual dense/projected triples sources,
primal identity and tile schedule. Do not trust matching caller-supplied labels.
The bound response is immutable, like the existing CCSD response: replacing its
corrected result, baseline or schedule after verification must not retain an old
response identity while changing the calculation. No extra solve runs per weight.

## Evidence and limits

Independent reconverged fixed-orbital finite differences remain the numerical
gate. Source-label and post-binding replacement regressions fail on the original
implementation and pass with checked immutable binding. CPU interpreter execution
remains explicit; this does not qualify GPU corrected Lambda, orbital response,
public nuclear forces or a complete endpoint-memory/performance guarantee.

Agent: ChatGPT
Model: GPT-6 Astra Pro
