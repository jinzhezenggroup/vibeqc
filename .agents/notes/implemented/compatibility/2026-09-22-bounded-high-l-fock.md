# Decision: bounded high l fock

Status: implemented
Date: 2026-09-22

## Decision

Expose the existing exact bounded contraction with Force=false for uncovered f-shell Fock classes. Exclude classes already emitted by the generated consumer to avoid double counting.

## Invariants and rejected alternatives

Keep explicit bounded storage and the existing basis/component and density conventions. A missing generated class must not silently lose a Fock contribution.

## Evidence and remaining qualification

The recovered tests and generation checks are retained. Current-source CUDA build, RHF/UHF high-l oracle comparisons and larger complete endpoints remain merge gates; historical evidence is not a fresh run.

## Revisit when

Revisit the candidate when the stated endpoint gates pass or the shared owner changes.

Fresh endpoint qualification and the additional order-three/profile fixes are
recorded in [the follow-up decision](2026-09-23-bounded-direct-endpoint-qualification.md).
