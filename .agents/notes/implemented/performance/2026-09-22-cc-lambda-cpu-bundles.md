# Decision: cc lambda cpu bundles

Status: implemented
Date: 2026-09-22

## Decision

Adopt shared TensorIR CPU bundle execution in native coupled-cluster and Lambda consumers, retaining explicit bundle ownership and existing mathematical programs.

## Invariants and rejected alternatives

Do not introduce a second CC algebra or silently change numerical precision. A consumer must keep the bundle and its input owners alive for execution.

## Evidence and remaining qualification

Focused CPU bundle, native tensor, Lambda solver and response suites passed (116 passed, 7 skipped). Complete endpoint work and memory measurements remain necessary for a performance claim.

## Revisit when

Revisit the candidate when the stated endpoint gates pass or the shared owner changes.
