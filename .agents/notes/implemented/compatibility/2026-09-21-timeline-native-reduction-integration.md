# Decision: preserve native source reduction during timeline integration

Status: implemented integration repair
Date: 2026-09-21

## Boundary

Current master reduces the complete seven-source all-electron gradient inside
the native stationary owner; scalar ECP still uses the external-source TensorIR
sum. Timeline integration must preserve both choices rather than restoring the
retired unconditional TensorIR final reduction.

## Decision

Apply the exclusive final-reduction phase to each existing branch. Keep metrics
collection separate. Account the reduced-output D2H call and its optional wall
attribution with the same rules as component output, without adding a new
synchronization or changing the authoritative reduction expression. Preserve
source-registry provenance and all unrelated current-master changes.

## Evidence and limit

The integrated timeline, cleanup, failure-evidence, reduction-authority,
merge-boundary, prepared-budget and schedule checks pass 63 cases. The source
registry verifies offline. These host checks do not claim a new complete device
timeline or performance matrix, and previous device measurements remain tied to
their recorded source versions.

Agent: ChatGPT
Model: GPT-6 Astra Pro
