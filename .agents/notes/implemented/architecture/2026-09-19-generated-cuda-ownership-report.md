# Decision: generate the current CUDA ownership report in CI

Status: implemented
Date: 2026-09-19

## Problem

The tracked current ownership JSON was a mechanically generated full-repository
snapshot containing source hashes, line counts, subsystem totals and ledger
copies. Any CUDA edit changed the file, so unrelated parallel PRs routinely
modified the same large JSON and conflicted after each merge to master.

## Decision

Keep docs/cuda_ownership.json as the reviewed semantic source of truth, but do
not version the mechanically derived current report. report_cuda_ownership.py
--check continues to fail closed on unclassified CUDA files, stale source
anchors, stale evidence/owners and inconsistent classifications. CPU CI also
writes .artifacts/cuda-ownership-current.json and uploads it as an artifact for
review and provenance.

Historical ownership baselines and benchmark-specific retained reports remain
versioned when their exact old bytes are part of a comparison contract.

## Rejected alternatives

A custom Git merge driver would not be reliably honored by GitHub server-side
merges. merge=union can create invalid JSON. Splitting the generated snapshot
into many tracked fragments reduces but does not remove mechanical churn and
still duplicates data derivable from source plus the semantic ledger.

## Invariants

- The semantic ownership ledger remains reviewed and tracked.
- New or removed CUDA files and stale or overlapping anchors still fail CI.
- Current ownership totals must be deterministic and internally consistent.
- Historical baselines used for physical/scientific delta claims remain immutable.
- Generated current reports are artifacts, never inputs that authorize a
  scientific retirement claim.

## Evidence

tests/python/test_cuda_ownership.py regenerates the report twice, checks
determinism, and validates report totals. Pre-commit runs
tools/report_cuda_ownership.py --check. The GCC CPU CI leg materializes and
uploads the current report.

## Consequences

Parallel CUDA PRs no longer all rewrite the same generated snapshot. Conflicts
are concentrated in real semantic ownership changes to the ledger or shared
source, where review is useful.

## Revisit when

Reintroduce a tracked current snapshot only if an external consumer requires a
stable repository path that cannot consume CI artifacts or regenerate the report,
and only with a conflict-free publication mechanism.

## References

Issue #231 and docs/cuda_ownership.md.

Agent: ChatGPT
Model: GPT-5.6 Sol
