# VibeQC agent instructions

These repository-local rules complement user-level agent instructions. They apply
repo-wide unless a nested `AGENTS.md` adds more specific constraints.

## Repository-wide invariants

- Prefer scientific correctness, reproducibility, and complete-endpoint behavior
  over isolated kernel or microbenchmark wins.
- Production paths must not silently depend on CPU/PySCF/reference-oracle work
  unless that consumer is explicitly part of the production contract.
- Performance claims must include complete endpoint timing and semantic work
  counts; memory-bounded is not necessarily work-bounded. See
  `docs/maintainer/performance_engineering.md`.
- Numerical, precision, derivative, and response changes require an independent
  oracle/reference and explicit acceptance gates appropriate to the method.
- Keep explicit bounded fallbacks when a faster path depends on optional resident
  storage, identity/lifetime assumptions, or other resource preconditions.
- Preserve durable rationale for non-trivial architecture, numerics, performance,
  and compatibility decisions as Agent Notes under `.agents/notes/`.

## Mandatory PR workflow

- Before implementation, continuation, review, or merge, read the current root
  `AGENTS.md` and applicable ancestor/scoped instructions. Fetch upstream and
  record the exact base/head, issue acceptance, existing changes, active owners,
  and running validation jobs. Use an isolated worktree per mutable workstream;
  do not overwrite another owner or restart a valid experiment to appear active.
- The implementer must inspect the complete diff, run the required validation,
  and publish an exact-head self-check with commands, results, findings, and
  limits. An author self-check is a `COMMENT`, never an independent approval.
- Every PR needs a substantive recorded review approving its exact final head
  by a reviewer who did not implement the change. Formally request an eligible
  non-author reviewer; an `@mention` can explain scope but is not a completed
  review. Record reviewer identity, head SHA, inspected scope, findings and their
  resolution, test/device evidence, and unverified limits. Do not impersonate
  another account or treat a second session as independent if it implemented
  the change. A bot name alone does not establish reviewer independence.
- For #1191 work, request `njzjz-bot` by default after self-check and required CI.
  If that reviewer authored or implemented the change, request another independent
  reviewer instead (for example, `njzjz`). Do not trigger `@codex review` or enable
  additional automatic review services without separate user authorization.
  This routing preference never waives the independence gate.
- A new commit requires delta re-review tied to the new head. Green CI, bot
  status, mergeability, a pending request, or an unsupported LGTM does not replace
  review. Scientific/precision/performance changes also need their applicable
  independent-reference, device, and complete-endpoint gates; compilation or a
  microbenchmark cannot stand in for those gates.
- Do not merge, enable auto-merge, or enter a merge queue before the review,
  required checks, and applicable acceptance gates pass and all blocking findings
  are resolved. Immediately before the authorized action, re-read the head,
  reviews, checks, target branch, and intended closing issues. Bind the expected
  head where supported. Use normal protected-branch procedures; never bypass
  checks, push directly to the default branch, or force-push shared work.
- A queue entry is not a merge. Observe the resulting merge SHA and verify issue
  closure. Partial work must reference its parent without closing it; in
  particular, infrastructure alone cannot complete #1190 or #1191. A finding
  discovered after merge needs an explicitly post-merge audit and corrective PR,
  not backdated approval. These rules grant no merge or release authority.

## Release authority

- Repository cleanup, benchmark evidence retention, fixes, PRs and merges do
  not authorize creating/publishing/editing/deleting GitHub Releases or release
  assets, creating/pushing release tags, or dispatching release/distribution
  publishing workflows. Each such operation needs explicit user authorization
  for that operation in the current task.
- Do not substitute a fork Release or another external host to bypass this
  boundary. Use existing Git history and ignored local artifacts for historical
  recovery; a new external backup requires separate approval.
- A recovery link, checksum, CI result, prior instruction to continue work or
  permission to open a PR is not authorization to publish a Release.

## Scoped instructions

Read the closest applicable nested instructions before editing:

- `docs/AGENTS.md` for current-state documentation versus historical rationale;
- `python/vibeqc_compiler/AGENTS.md` for compiler ownership and generation rules;
- `src/integrals/AGENTS.md` for integral, derivative, precision, and scheduling
  constraints; and
- `.agents/notes/AGENTS.md` before adding or revising an Agent Note.

## Agent Notes

`docs/` describes the current system: what is true now and how to work with it.
`.agents/notes/` preserves why durable decisions were made, including discarded
alternatives, measured evidence, and conditions for revisiting them.

Add a note for a non-trivial change when it changes architecture/ownership,
scientific numerics or precision, algorithmic work/data movement, or a retained
compatibility/fallback policy. Ordinary bug fixes, tests, and local mechanical
refactors do not need notes. As a practical rule, if a diagnosis took substantial
investigation and a future agent could plausibly repeat a failed approach, record
it.
