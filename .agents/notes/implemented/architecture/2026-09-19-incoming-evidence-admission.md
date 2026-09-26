# Decision: Review incoming benchmark evidence separately from checkout capacity

Status: implemented
Date: 2026-09-19

## Problem

#488 identified renewed growth after #238. At source `5a7fdeb2689553c0a304dad3338ba184d850ef60`,
the results checkout contains 2,344 tracked files and 105,281,931 bytes. A per-file
limit admits arbitrarily many small dumps; a near-full aggregate cap repeatedly
blocks unrelated development. Most primary runners already use `.artifacts/`,
so merely changing their defaults again would not fix admission of run output.

## Decision

Keep aggregate capacity (including the independent #518 adjustment) separate
from the new-change review budget. Count complete added/modified evidence blobs
against 2 MiB; deletions provide no credit. CI compares the staged integration
tree with the explicit PR base, and all existing full-tree guards still apply.
Hash-pinned policy reasons can explicitly waive this review budget for required
scientific inputs, but cannot waive the hard per-file/aggregate checks.

Recognize full profiler arrays by content, not only filenames. Require exact-byte
retention justification for newly introduced nonempty `launch_records` and
`traceEvents`. Do not automatically reject scientific arrays or negative results.

Add an auditable per-file/per-family inventory that distinguishes byte-bound
publications and justified exceptions from unclassified legacy data. A coarse
location-based label is not a claim of review. Nothing in this inventory
authorizes deletion or certifies numerical/performance conclusions.

Guard the shared benchmark writer and six consumers against raw output into the
reviewed results checkout, including symlink aliases. Retention publication is a
separate explicit step. Other historical writers remain follow-up work.

## Rejected alternatives

- Repeating ad hoc historical cleanup whenever capacity is exceeded does not
  control new evidence and risks losing test or production qualification inputs.
- Net byte growth permits an unrelated deletion to subsidize a new large dump.
- Blanket JSON/NPZ/ZIP bans confuse scientific replay inputs with transient data.
- Treating all files under `results/` as reviewed conceals the remaining audit.
- Duplicating scientific validation in the storage guard changes ownership.
- Releases or new external backup locations are not authorized by this work.

## Invariants and evidence

This slice changes no production/compiler code, scientific fixtures, existing
result bytes, numeric thresholds or benchmark measurements. Tests cover staged
versus unstaged content, deleted/renamed files, exact-hash exceptions, raw JSON
markers, malformed/incomplete publication inventories, explicit Git bases,
offline object reads, and writer/path safety. No native/GPU performance claim
is made. See `test_evidence_change_review.py`, `test_benchmark_output_retention.py`
and the existing retention/publication tests.

## Consequences and revisit conditions

A large new justified evidence set needs reviewable policy rationale rather than
silent admission. PRs must have their base objects locally; checks do not fetch.
The campaign inventory exposes manual review work instead of declaring it done.
#488 stays open for a consumer-aware audit, any justified historical migration,
and conversion of the remaining independent writer paths. Revisit the 2 MiB
review threshold with measured maintenance evidence, not automatic inflation.

Refs #488, #238, #516, #518.

Agent: ChatGPT
Model: GPT-6 Astra Pro
