# Decision: reduce benchmark checkout size using existing Git history

Status: implemented
Date: 2026-09-19

## Problem

Many individually small historical reports defeated the existing 1 MiB per-file
budget. Original PR #474 used an unauthorized Release as its archive, and the
subsequent single-file Git fallback did not establish complete-snapshot recovery.
The [withdrawn proposal](../../rejected/2026-09-19-unapproved-benchmark-release.md)
is retained as rejected history, not an active storage dependency.

## Decision

Remove the same audited 197 bulky files (40,562,398 bytes) from checkout, while
preserving all 917 original campaign members (55,852,909 bytes) in the existing
ancestor `e215b30685f8a36ef0cc5772c9166837527f64a2`. The Git snapshot manifest
records each original path, size and SHA-256, distinguishing checkout removals
from retained paths. Extend the existing restoration helper with `--all` rather
than adding another storage service or archive format. Verify every member in
temporary local storage before creating a new full-snapshot destination.

Keep concise summaries, comparison samples, reproduction scripts, source patches,
input arrays and all test/production consumers in Git. Preserve the legacy
single-file migration/ZIP-manifest interfaces and existing compact archive tests.
Retain the reviewed 96 MiB aggregate results budget and unchanged 1 MiB file cap.

## Invariants

No Release, release tag/asset, external upload or publishing workflow is required
or authorized. No implicit Git fetch (including partial-clone lazy fetching),
execution of historical scripts, overwrite of existing output, or Git-history
rewriting occurs. Missing history produces an explicit user-controlled recovery
instruction. Rejected/incomplete runs and numerical values remain unchanged.
Local evidence preparation is not permission for external publication.

## Rejected alternatives

Republishing a Release on upstream/a fork or choosing another host repeats the
authorization failure. Expiring CI artifacts are not a durable replacement for
required evidence. Another large tracked ZIP defeats checkout cleanup. Deleting
original Git history defeats this recovery contract. Keeping only single-file
recovery is insufficient for validating original complete campaign manifests.

## Consequences and validation

Current checkout size shrinks, but full-clone historical size does not. Source
archives without Git metadata cannot restore old objects; shallow/partial clones
may need an explicit fetch before offline recovery. Administrators can remove
history, so hashes guarantee identity rather than unlimited availability.

Tests cover exact text/binary restoration, late-member corruption or absence,
unsafe/duplicate/conflicting paths, shallow/missing-history behavior, refusing
existing output and legacy compatibility. Verify all 917 real snapshot members
and the exact 197-file removal set before submitting this change. Ordinary CI
uses tiny local Git fixtures and does not download historical data.

## References

- #238 and #474: repository retention/checkout cleanup.
- [Current policy](../../../../docs/evidence_retention.md).
- [Snapshot and recovery](../../../../benchmarks/results/retention-checkout/README.md).

Agent: ChatGPT
Model: GPT-6 Astra Pro
