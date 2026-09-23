# Decision: project the bulk runtime ABI without rewriting the import

Status: implemented
Date: 2026-09-23

## Problem

PR #1126 compacted optional meta-GGA inputs by passing zero constants into the
Maple worker when registration flags omitted those ingredients. Flags alone do
not prove that the imported expression is independent of a feature. A mismatch
could silently change the expression while retaining a pointwise-qualified label.
Changing the canonical importer for runtime packing also changed its source hash
and broke the offline source-registry gate for the otherwise unchanged catalog.

## Decision

Keep the full existing bulk import, including its original feature variables,
Graph and source identity. At the runtime boundary select the declared feature
families and require every discarded input to be unreachable from the energy
root. Reject any disagreement; do not specialize a live input to zero. Differentiate
only retained variables. Runtime expression identity already binds the selected
features, outputs and emitted graph independently of the import identity.

Restore libxc_bulk.py byte-for-byte to its existing canonical blob
7e1c8193511db502be792d57b20002777f3d6653 (SHA-256
6af8dc2b3c21ec49cf003af2d842fedd060a34a5302c1e01f4f75c31928f1967).
No registry assertion, source manifest, retained reference or tolerance is removed
or rewritten. The experimental compact_features importer argument introduced
only by this unmerged PR is not needed by the runtime-facing API.

## Evidence and limits

Ten focused local host tests pass using the fetched runtime module and existing
packaged compiler/Libxc assets, not a complete current repository checkout.
Four injected laplacian/tau flag disagreements across both spin layouts fail to
reject with the original PR importer/runtime, and reject after repair. Six
real MGGA_X_R2SCAN01 cases preserve full-import source identity and compare
energy/vxc/fxc with the full Graph at nonzero signed Laplacians. These are
projection-equivalence checks, not an independent physical oracle. The existing
independent bulk fixtures and shared CUDA emission tests remain unchanged and
must pass current-head repository CI. No new runtime/public admission or GPU
qualification is claimed.

## Revisit when

A future importer specialization has independently qualified semantics and a
separate provenance contract, or a new runtime ingredient ABI is admitted.

Agent: ChatGPT — Even-PR Review
Model: GPT-6 Astra Pro
