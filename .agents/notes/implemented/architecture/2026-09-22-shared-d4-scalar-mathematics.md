# Decision: shared D4 scalar mathematics

Status: implemented
Date: 2026-09-22
Updated: 2026-09-25

## Decision

Use one handwritten D4 scalar-math owner for charge scaling and reference weights, coordination values and derivatives, damping values and derivatives, damping radii, and C6 interpolation derivatives. The retained fixed-charge evaluator and the live GFN2 CPU/CUDA consumers call this owner directly.

The GFN2 runtimes continue to own topology, pair-list/cache traversal, cutoff admission, validation, reductions, failure semantics, and publication. CPU plan construction keeps precomputed coordination and damping parameters so the ownership cutover does not add repeated element-table work to the hot geometry loop.

## Invariants and rejected alternatives

This consolidation does not promote a new D4 production schedule, move GFN2 runtime ownership into the DFT evaluator, or relax any SCC/CUDA validation and cutoff contract. The CUDA consumer validates atomic numbers, reference bounds, Gaussian counts, charges, coordination numbers, and finite shared-helper outputs before publication.

Do not restore private GFN2 copies of the shared scalar formulas merely to preserve the upstream source layout. If runtime-specific scheduling needs precomputed values, add a shared primitive that accepts those precomputed values instead.

## Evidence and remaining qualification

The live CPU adapter storage/publication suite passes 10/10. The full `vibeqc_gfn2_cuda` archive compiles and device-links for SM120, including the live `gfn2_d4.cu` consumer. Independent CPU/CUDA fixed-charge D4 gates match dftd4 at about 1e-18 to 1e-19 maximum error, and CUDA ragged-batch/peer-isolation checks pass. Source-ownership and CUDA provenance tests cover the live cutover.

Complete GFN2 SCC/CUDA/periodic parity and endpoint performance remain required before the PR is marked ready.

## Revisit when

Revisit this decision if the scalar owner changes, a runtime needs a new precomputed primitive to preserve endpoint performance, or qualification exposes a semantic difference in charge/CN boundary handling.
