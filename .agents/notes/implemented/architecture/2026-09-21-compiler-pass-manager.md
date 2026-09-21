# Decision: one deterministic pass manager for compiler optimization pipelines

Status: implemented
Date: 2026-09-21

## Problem

TensorIR already had a useful sequence of conservative rewrites, but the order was encoded directly in `tensor.optimize`. As additional IRs gain canonicalization, DCE, scheduling cleanup, and target-aware optimization, copying ad hoc loops would make pass ordering, versioning, diagnostics, and bisection inconsistent. Issue #682 requires a common optimizer pipeline without moving scientific policy into `common`.

## Decision

Add a backend- and IR-neutral `common.pass_manager` contract. Each stage has a stable name/version, transform, optional prerequisite stages, and explicit analysis invalidation labels. A manager validates dependencies, runs stages deterministically, supports disabling stages or stopping at a prefix for bisection, records before/after fingerprints, and derives a content-addressed identity from the active versioned pipeline.

TensorIR is the first consumer. Its existing rewrite implementations and exact order remain unchanged; only orchestration moves through the common manager. TensorIR preserves the established `rewrites` provenance field and additionally records optimizer identity plus compact per-stage change diagnostics.

## Rejected alternatives

- Keeping independent pass loops in every IR was rejected because pass bisection and optimizer identity would immediately diverge.
- Moving TensorIR rewrite mathematics into `common` was rejected because the common layer should own orchestration, not IR-specific semantics.
- Building a full analysis-cache framework in this first slice was rejected as premature. Invalidation is explicit in stage metadata/trace; analysis ownership stays with each IR until multiple real consumers require shared caching.
- Inferring pass changes from object identity was rejected because immutable rewrites may reconstruct equivalent objects. Consumers provide deterministic structural fingerprints instead.

## Invariants

- Pass order and versions are deterministic and participate in pipeline identity.
- Unknown disabled/stop stages and unmet required stages fail closed.
- Scientific rewrites remain owned by their IR packages.
- Disabling a required prerequisite cannot silently run a dependent pass.
- TensorIR mathematical identity and rewrite order are unchanged by this orchestration refactor.

## Evidence

Focused pass-manager and TensorIR execution tests cover deterministic replay, changed/no-op diagnostics, invalid dependencies, bisection, and unchanged logical equations. Compiler structure and Ruff checks are part of the PR validation.

## Consequences

Future #682 slices can add canonicalization, GVN/CSE, post-schedule cleanup, and target-aware resource passes through one versioned pipeline while retaining IR-specific transformations. Analysis caching can be layered onto the explicit invalidation contract when a second consumer demonstrates the need.

## Revisit when

Introduce an explicit shared analysis cache only when at least two IR consumers need the same analysis lifecycle. Extend identity semantics if a pass has target/toolchain parameters that are not already represented by its version or surrounding artifact identity.

## References

- #682
- #673
- `.agents/notes/implemented/architecture/2026-09-20-shared-liveness-dce.md`

---
Agent: ChatGPT
Model: GPT-5.6 Sol
