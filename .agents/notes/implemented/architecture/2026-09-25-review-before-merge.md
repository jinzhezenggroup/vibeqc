# Decision: exact-head independent review before merge

Status: implemented
Date: 2026-09-25

## Problem

Parallel issue/PR work can confuse author self-checks, CI success, review requests,
queue admission, and actual independent approval. PR #1306's campaign-preflight
infrastructure merged before a review-requested source-contract integrity fix;
#1308 is the separate corrective change. Infrastructure completion also must not
be confused with #1190's frozen scientific/performance acceptance.

## Decision

Keep the operational workflow in root `AGENTS.md` so every implementation,
continuation, reviewer, and merge coordinator reads the same rules. Require a
substantive independent review of the exact final head before merge, auto-merge,
or queue admission, with delta re-review after new commits. Preserve author
self-checks as comments, without relabeling them as independent approval.

A reviewer request or mention is not completion. Select a reviewer who did not
implement the change; a different session or bot account name alone is not proof
of independence. Keep source-matched tests and method-specific scientific gates
separate from the review decision. Recheck head, blockers, and closing issues at
the action boundary and verify actual merge/closure afterward.

For the user-directed #1191 workflow, the default formal reviewer is
`njzjz-bot`, with a different independent reviewer when that account owns the
implementation. Do not initiate `@codex review` or enable additional review
services without separate authorization; this records the user's routing
preference, not any inference about service billing or account configuration.

## Rejected alternatives

CI-only merging misses semantic and acceptance-contract defects. Automatically
carrying an old approval across new commits leaves unreviewed changes. Requiring
a particular bot account for every PR would also fail when that account owns the
implementation. Rewriting history with post-hoc approval hides rather than
repairs a missed gate.

## Invariants and consequences

These instructions do not grant merge, release, or branch-protection bypass
permission. Independent scientific evidence, precision thresholds, and finite
product scope remain unchanged. Queue delays and real blockers are not coding
stalls; preserve work and avoid duplicate owners. A policy-only PR is not product
acceptance and must itself follow the review gate.

## Evidence and references

- User-directed #1191 coordination and review-before-merge requirement.
- #1306: campaign preflight only; scientific acceptance remains not run.
- #1308: complete frozen-contract integrity repair following the merge race.
- #1190: overall PASS on one exact merged production revision is the final gate.

## Revisit when

A separately reviewed workflow change provides equally auditable exact-head
independence and safely coordinates protected merge queues. Do not silently
weaken the gate merely to clear a pending queue or a failed validation.
