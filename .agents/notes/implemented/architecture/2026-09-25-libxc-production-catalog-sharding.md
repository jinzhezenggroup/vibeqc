# Decision: shard broad Libxc production-domain campaigns deterministically

Status: implemented
Date: 2026-09-25

## Problem

The single-functional B1 runner can produce exact receipts, but manually invoking
it for hundreds of imported registrations does not scale and makes it easy to
lose negative results or accidentally retest an arbitrary subset. A broad B2
campaign needs deterministic partitioning, per-functional retained outputs, and a
compact blocker inventory without treating the report itself as admission.

## Decision

Add a catalog campaign that starts from the canonical imported capability
inventory sorted by registration name.

- Optional name filters are case-insensitive and reject unknown registrations.
- Shards use the stable sorted index modulo an explicit shard count.
- Each eligible registration runs the existing single-functional campaign and
  writes one JSON file.
- Structurally blocked registrations are summarized without fabricating a
  production-domain receipt.
- Runner errors remain explicit rows rather than disappearing from the inventory.
- The shard summary records pass/fail/not-run/structural-blocked/runner-error
  counts, family counts, and blocked matrix-case counts.
- A catalog identity binds selected capability/profile identities, shard
  parameters, candidate domain, and numerical tolerances.
- Negative results do not fail the collection command unless
  `--require-all-pass` is selected.

Raw campaigns are directed to scratch/artifact output. Review/publication remains
a separate evidence-retention step.

## Rejected alternatives

- One monolithic JSON for every functional would be difficult to review and
  likely exceed evidence-change budgets.
- Parallel workers choosing names ad hoc would make campaign coverage
  irreproducible.
- Omitting failed/blocked registrations would bias the apparent support matrix.
- Automatically committing all raw campaigns would conflate transient evidence
  collection with reviewed retention.

## Invariants

- Shards are deterministic, disjoint, and reconstruct the selected inventory.
- Every eligible campaign retains its exact receipt/execution identity.
- Structural blockers never receive forged receipts.
- Reports do not promote capability.
- Negative results are first-class evidence.

## Evidence

Tests cover complete/disjoint four-way sharding, case-insensitive filtering,
invalid shard parameters, deterministic status/family/blocker counts, per-file
campaign retention, and structural blocker reporting.

## Consequences

B2 can now run the full imported catalog on parallel workers and immediately
identify which boundary case classes dominate failures. A later aggregation or
review step can select compact accepted/negative evidence without rerunning the
scientific campaign.

## Revisit when

The imported inventory ordering changes deliberately, distributed scheduling
needs stronger work balancing than deterministic name sharding, or reviewed B2
evidence gets a dedicated publication schema.

## References

- #1118
- #1120
- #1315
- #1320
- #1322
- #1325

Agent: ChatGPT
Model: GPT-5.6 Sol
