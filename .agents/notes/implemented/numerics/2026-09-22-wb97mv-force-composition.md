# Decision: wb97mv force composition

Status: implemented
Date: 2026-09-22

## Decision

Compose the WB97M-V semilocal, range-separated and nonlocal response through KS snapshots and stationary CPU force resources. Regenerate method metadata from the manifest while retaining current semantic fields.

## Invariants and rejected alternatives

Energy admission does not establish analytic force qualification. Snapshot lifetime, grid/atomic weights and spin semantics must agree across native and Python owners. Keep this candidate in draft until those contracts and the complete endpoint pass.

## Evidence and remaining qualification

The current-source CPU library builds and native self-consistent composition tests pass. Python diagnostics map native domain version 3 explicitly, and canonical manifest aliases share one method/spin binding. Multi-step finite differences, force/translation invariants and public capability/alias consistency remain explicit merge gates.

## Revisit when

Revisit the candidate when the stated endpoint gates pass or the shared owner changes.
