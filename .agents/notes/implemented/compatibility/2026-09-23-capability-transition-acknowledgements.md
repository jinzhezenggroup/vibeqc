# Decision: bind capability exceptions to exact transitions

Status: implemented
Date: 2026-09-23

## Problem

A kind/stage/functional-only token remains identical when the scientific source
changes again. An exception approved for identity-v1 -> identity-v2 would also
approve identity-v2 -> identity-v3, defeating stale-exception rejection.

## Decision

Append the shared canonical hash of the functional's validated previous/current
identity and qualification state to its readable regression token. Normalize
stage ordering and exclude unrelated functionals. Retain an immutable digest in
the returned regression rather than borrowing mutable snapshot dictionaries.

## Rejected alternatives and invariants

Do not retain legacy name-only tokens as aliases: that restores the bypass.
Do not hash the whole catalog: unrelated additions would invalidate a deliberate
exception. Reuse existing snapshot validation, change classification and hashing;
no evidence is created and no scientific/public capability is promoted.

## Evidence and consequences

Five cross-transition token-replay cases and one changed-snapshot identity case
fail before the repair. The repaired 12-case host slice passes using the current
transition code, isolated current snapshot validator/change detector and stage
DAG, and the byte-matched shared hash helper. Full repository CI remains the
integration gate; this is not a native/GPU numerical qualification.

Consumers obtain a new explicit token for each intended state transition. Revisit
only when the snapshot schema or the scope of an acknowledged exception changes.

Refs #1142, #1124.

Agent: ChatGPT — Even-PR Review
Model: GPT-6 Astra Pro
