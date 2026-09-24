# Decision: Generated-source response owner and batch fallback

Status: implemented
Date: 2026-09-21

## Problem

Generated CUDA DF plans release the host three-center tensor after setup. A
resident response route therefore cannot infer a raw `A` owner from an empty
host span. Retaining only whitened `B` is insufficient for exact metric
response, especially when the metric has discarded directions. Batch plans
also cannot publish one singleton raw owner for every response item.

## Decision

For dense generated plans, setup gathers the complete generated pair-major raw
tile into the existing auxiliary-major `exchange_contributions` allocation
before whitening. The owner is published only for a resident singleton plan
with complete AO and auxiliary layout and a valid 32-bit launch shape. The
same copy is performed for full-rank and rank-deficient metrics so discarded
directions remain available to the response.

When a full resident source plan has no persistent raw owner, response may
regenerate the current source item's complete raw tensor into the borrowed
full-size buffers for that response only. This keeps batch execution correct
without pretending that a batch plan has a reusable singleton owner. Partial,
streamed, UHF, corrected-density, and incompatible ownership cases retain
their bounded fallback routes.

## Invariants

- `resident_raw_valid` is set only after every raw tile and whitening operation
  succeeds.
- The raw owner is auxiliary-major `[Q, mu, nu]`; generated source tiles remain
  pair-major until the existing gather kernel converts them.
- Raw reuse requires the same metric/source owner and never uploads the raw
  tensor.
- A response regenerated for a batch item is not published as a cross-item
  cache.

## Evidence

- CPU DF resource and budget tests: `99 passed, 2 skipped`.
- CUDA source reuse, occupied response, and resident response regression set:
  `52 passed` on RTX 5090 / sm_120 / CUDA 12.9.1.
- CUDA release build completed successfully.

## References

- Issue #206.
- `src/scf/cuda/df_plan_setup.cpp`.
- `src/scf/cuda/df_gradient_bridge.cu`.
