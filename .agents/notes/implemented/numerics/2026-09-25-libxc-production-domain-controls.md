# Decision: qualify Libxc production controls as shared invariants

Status: implemented
Date: 2026-09-25

## Problem

The v2 production-domain matrix contains two control rows that are not physical
XC coordinates: lazy inactive-branch semantics and invalid/nonfinite input
rejection. Treating them as numerical oracle points is category error. Leaving
them permanently `not-run`, however, prevents every otherwise-correct imported
functional from ever completing the exact matrix.

## Decision

Own these rows as shared compiler/runtime control evidence.

For `control/invalid-nonfinite`, build the exact bulk runtime program for the
functional/spin layout and inject NaN, +infinity, and -infinity into every
declared feature position. Every case must be rejected by the runtime feature
validator with the explicit nonfinite-domain reason before Graph evaluation.

For `control/lazy-inactive-branch`, use the shared `Graph.select_le`
primitive with a deliberately singular inactive branch `1/x`. At the selected
endpoint x=0, validate the value, first derivative, and second derivative, the
lane-wise array interpreter, and both scalar-C and CUDA lexical branch emitters.
The singular branch remains present in emitted source but must not be evaluated.

These checks are generic rather than functional-specific. Their evidence is
nevertheless attached inside each exact functional receipt, whose capability
identity already hashes the common/integral/XC compiler sources.

## Rejected alternatives

- Inventing fake rho/sigma/tau coordinates for control cases would blur
  mathematical qualification with runtime/compiler invariants.
- Marking controls passed solely because unit tests exist would not bind the
  result to the evidence campaign or functional source identity.
- Removing the control rows would weaken the production-domain contract and
  permit future eager-branch or nonfinite-input regressions to bypass admission.

## Invariants

- Nonfinite inputs fail before mathematical evaluation.
- Inactive piecewise branches are not evaluated by scalar or array execution.
- Symbolic first/second differentiation retains the same lazy predicate.
- CPU and CUDA emitted source retain lexical control flow for the branch.
- Passing a control row grants no numerical boundary, SCF, force, response, or
  public-method capability by itself.

## Evidence

Focused tests run both controls for representative non-curated LDA, GGA, and
tau-MGGA registrations in both spin layouts. Additional tests lock second
derivative behavior, C/CUDA branch emission, per-feature nonfinite rejection,
and fail-closed handling of non-control/cross-spin requests.

Repository CI is the executable authority for the stacked branch.

## Consequences

The B1 campaign can now produce real pass evidence for both generic control rows.
Remaining production-domain blockers are numerical boundary cases rather than
schema/control placeholders.

## Revisit when

The Graph piecewise primitive, runtime feature-validation ownership, or
production-domain control taxonomy changes.

## References

- #1118
- #1120
- #1314
- #1315

Agent: ChatGPT
Model: GPT-5.6 Sol
