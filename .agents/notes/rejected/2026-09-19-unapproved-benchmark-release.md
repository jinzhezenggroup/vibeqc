# Decision: move bulky historical reports out of the source checkout

Status: rejected
Date: 2026-09-19

This withdrawn, unmerged design is preserved only to explain why it must not
be repeated. Its Release was unauthorized and removed; the storage and restore
claims below are **not current behavior**. Do not recreate that Release.
Superseded by [Git-only snapshot recovery](../implemented/architecture/2026-09-19-benchmark-checkout-retention.md).

## Problem

The `e215b30685f8a36ef0cc5772c9166837527f64a2` tree contains 156,582,232
tracked bytes, of which 139,677,516 are under `benchmarks/results/`. The
existing 1 MiB per-file cap admits many individually small historical reports.

## Decision

Move an audited set of 197 bulky reports/archives (40,562,398 bytes) out of
18 historical DF campaign directories. Preserve all 917 original campaign
members in a 10,518,603-byte upstream release asset, with a source-pinned,
per-member hash/size manifest. Verify the locally generated archive and then
download and restore the uploaded asset before removing any original file.

Keep concise summaries, reproduction/source patches, original input arrays,
all test-consumed bundles and all production-referenced evidence in Git.
Reuse the existing archive verifier and extend the existing Git restoration
entrypoint with an optional manifest; do not introduce a new evidence service.
Normal tests and builds remain offline. Preserve rejected/incomplete runs and
all original numerical/statistical values in the snapshot.

Add a reviewed 96 MiB aggregate benchmark-results budget to the existing
Git-index retention check. Permanent test fixtures are outside this aggregate
benchmark budget. Do not alter the existing per-file cap.

## Rejected alternatives

Deleting raw evidence without recovery loses auditability. Committing another
large ZIP merely hides diffs and violates retention policy. Expiring CI assets
are not appropriate as the only durable copy. Rewriting Git history would
disrupt forks and branches and is outside this change. Moving test-consumed
fixtures would add an unwanted network dependency to ordinary validation.

## Invariants and consequences

Archive manifests describe original source-revision bytes, not future revisions
of retained summary files. Historical claims keep their original source,
limitations and gates; cleanup does not requalify them. Existing ancestor Git
objects are an independent recovery route, not a policy to commit future raw
runs. The release has no automatic expiry, but hosting and Git objects can
still be removed by administrators. Checksums establish byte identity, not
scientific validation.

Current checkouts become smaller; full-clone history is unchanged. Further
reduction requires auditing the remaining direct test/production consumers,
not mechanically deleting by extension or size.

## References

- #238: durable evidence versus transient artifacts.
- [Current policy](../../../docs/evidence_retention.md).
- [Archive and recovery](../../../benchmarks/results/retention-checkout/README.md).

Agent: ChatGPT
Model: GPT-6 Astra Pro
