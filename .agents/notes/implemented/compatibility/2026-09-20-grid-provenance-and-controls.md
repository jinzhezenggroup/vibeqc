# Decision: own snapshot provenance and distinguish grid controls

Status: implemented
Date: 2026-09-20

## Problem

The #615 grid snapshot used write-once attributes but exposed a mutable provenance
dictionary. Callers could alter the source labels used by the model identity.
The convergence benchmark also called a v2 element-radius grid `pbe-legacy48`,
although the historical public default was a v1 unit-radius `GridSpec()`.

## Decision

Copy snapshot provenance into a read-only mapping, retaining `None` for legacy
CUDA snapshots. Convert to a plain dictionary only at the identity serialization
boundary. Cover caller aliasing and item mutation as well as reassignment.

Keep the 48-point v2 grid as a radial-count control, under `pbe-radial48`, and add
separate actual v1 controls for LDA and PBE. Bump the evidence schema to version 2
because the control names changed. Do not change production grids or gates.

## Historical evidence and remaining boundaries

The archived `benchmarks/results/issue596-production-grid/grid_policy_convergence.json`
is schema version 1. Its `pbe-legacy48` row measures the element-aware v2 radial
control, not the historical unit-radius default. Preserve those original bytes
and measured results; do not relabel them as a newly measured v1 endpoint.

Point-count ratios and PySCF oracle timings are not native VibeQC endpoint
performance. Native energy/force timings and real-NVIDIA qualification must be
reported separately with source identity; no new passing result is asserted here.

## Evidence

The owned-mapping patch passed 21 provenance/r2SCAN compatibility cases on node3.
The numerical benchmark extension requires a fresh independent-oracle run.

Agent: ChatGPT
Model: GPT-6 Astra Pro
