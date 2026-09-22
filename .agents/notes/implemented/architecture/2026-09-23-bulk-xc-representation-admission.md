# Decision: derive XC representation from qualified bulk capabilities

Status: implemented
Date: 2026-09-23

## Problem

A manually extended component whitelist cannot express the existing audited bulk
Libxc inventory without conflating representability with molecular qualification.
The shared native lowerer and runtime now use the same expression dispatcher,
so an admission check confined to the old runtime helper would be ineffective.

## Decision

Use the pinned bulk registration family and ingredient flags to admit representable
FunctionalSpecs. Exclude every registration requiring a Laplacian because the
current feature contract contains only rho, sigma, and tau. Preserve the curated
named method catalog, and label automatic component provenance as pointwise-only.

Reject nonzero automatic components in the shared expression dispatcher before
constructing a production graph. Both runtime programs and native CPU/CUDA source
lowering therefore require separate domain qualification. A zero coefficient does
not change the active family or promote its capabilities. Existing curated
components keep their original expressions and production policy.

## Rejected alternatives

Extending the curated public catalog would grant unintended method aliases.
Treating source emission or pointwise fixtures as SCF/force qualification would
skip required endpoint evidence. Retaining the old runtime-only admission helper
would leave the newer shared lowering outside the explicit qualification gate.

## Invariants

Capability evidence remains bound to the actual pinned inputs and compiler source.
Representation never implies production, GPU execution, force, or public admission.
New ingredients must be modeled explicitly before their registrations can enter
the automatic representation inventory.

## Evidence

The inventory contains 213 representable registrations and eight Laplacian
blockers; 197 represented names are outside the curated component tuple. Tests
check both spin layouts across the complete automatic inventory, and native
lowering rejects it in both ordinary and production modes. Existing independent
Libxc fixtures continue to qualify the separate pointwise importer.

## Revisit when

A Laplacian feature primitive is added, or a bulk component gains independently
qualified production-domain lowering and complete molecular endpoint evidence.

## References

PR #1037; bulk capability and source-identity contracts in
`python/vibeqc_compiler/xc/libxc_bulk_capabilities.py`.
