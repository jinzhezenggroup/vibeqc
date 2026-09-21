# Decision: reserve meta constants and reject aliased differentiation placeholders

Status: implemented
Date: 2026-09-21

## Problem

Review of #769 found two silent semantic failures. MU_GE, K_FACTOR_C and
DBL_EPSILON were recognized before global assignments and external bindings,
but their names were not reserved. A supplied value was accepted and ignored.
The bounded eval(diff(...)) translator also admitted the same placeholder in
both argument positions while lowering only one partial derivative.

## Decision

Reserve the three names at global source and external-binding admission, as for
Pi and X2S. Preserve function-parameter shadowing through the existing lexical
environment. Reject aliased differentiation placeholders rather than implying
support for an unimplemented multivariate chain rule. Distinct placeholders may
still be substituted with the same physical expression after differentiation.
Advance the importer semantic identity to v5; source and parameter identities
remain separately bound. No functional equation or numerical tolerance changes.

## Evidence and limits

An exact-head wheel module was verified against Git blob
`e49ee04cf96f90d3c5b6877e22eb4d0636ff8a6e`. Ten new admission cases fail before
repair and all fifteen focused host scalar cases pass afterward, including
lexical parameter values and derivatives and both valid partial derivatives.
The published repaired blob matches the tested source exactly. The earlier
38-test family selection passed before these extra cases were added; its final
integration rerun remains a CI gate, not a fresh GPU or production qualification
claimed by this repair. #769 merged concurrently during review, so this is a
focused follow-up rather than an approval of its original head.

Agent: ChatGPT
Model: GPT-6 Astra Pro

## Integration with signed-index repair

The later integration retains the already-merged #797 negative bounded-index
parenthesization and constant reservations. The distinct-placeholder guard joins
those semantics under importer v7; v5/v6 are not reused for this combined behavior.
Both regression families remain in the final source tree.

Agent: ChatGPT
Model: GPT-6 Astra Pro
