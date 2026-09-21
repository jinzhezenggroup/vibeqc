# Decision: reconcile the Libxc registry at integration without changing products

Status: implemented
Date: 2026-09-21

## Problem

Independently merged Maple qualification and source-registry work left the
combined tree with an incomplete LYP source inventory and an older importer
identity. Passing branch-only tests did not validate GitHub's merged test tree.

## Decision and invariants

Integrate the actual mainline source, then register the already-pinned LYP C
parameter owner using its existing manifest URL and verified content hash.
Bind the importer version/content and the admission-only product to the reviewed
combined implementation. Track the local PBE adapter when that adapter exists.
The admission product has no numerical outputs: do not re-bless generated tables,
change upstream revisions, weaken freshness checks, or overwrite reference data.
All four derived manifests must render byte-for-byte identically after repair.

## Evidence

The existing registry regressions fail before reconciliation and pass afterward.
The offline verifier checks every registered source and numerical output as well
as the repaired dependency identities. Functional-specific tests remain separate
acceptance gates; metadata reconciliation is not a new molecular/GPU result.

Agent: ChatGPT
Model: GPT-6 Astra Pro
