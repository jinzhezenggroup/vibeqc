# Decision: generated, tiled Becke-grid directional response

Status: implemented (CPU diagnostic building block; no public force capability)
Date: 2026-09-19

## Problem

Merged #455 completed #163 A, including the exact SCF-domain XC coefficient
bridge and generated AO-jet pullback. Its point and weight partials still needed
a physical grid-motion source. The existing molecular grid supplied values only;
repeating A or writing another handwritten PBE force would not fill that gap.

## Decision

Reuse the common scalar Graph for local norm, ratio, repeated Becke-switch and
log JVPs. Stream raw atomic tiles from one owner shared with the existing value
path, translate points by their owner displacement, and reduce all partition
center effects before normalizing. Interpret generated scalar roots on CPU in
this slice. Keep native state validation and complete stationary assembly outside
the compiler utility. The geometric consumer lives in the `xc` layer beside
existing XC geometric contractions: that layer already composes low-level `dft`
quadrature with the shared `integral.expr` scalar algebra. The lower-level `dft`
layer retains the raw/value-only grid and does not acquire an upward dependency
or a second copy of the scalar graph. No dependency-checker exemptions are added.

Keep log-domain products, with a separate zero-factor count and tangent. A product
with exactly one zero factor can have a nonzero derivative: the derivative of
that factor times all the other factors. Therefore do not replace every such
product derivative by zero or evaluate `0 * (dp/p)`. With multiple zero factors,
the first derivative is zero. This is the product's boundary rule, not a new XC
formula or a clipping policy that changes the primal energy.

A frozen common scale in the norm JVP is valid by homogeneity; it avoids unscaled
coordinate squares without adding an arbitrary distance floor. The maximum-log
product shift is likewise a common normalization scale: its derivative cancels
from the final quotient. Pair saturation branches remain inspectable.

## Rejected alternatives

- Handwritten Becke derivative polynomials: duplicate the shared algebra and make
  later native lowering harder to audit.
- Full coordinate-by-grid Jacobians or regeneration per nuclear coordinate:
  unnecessary storage/work for a directional consumer.
- Unnormalized direct products for ordinary evaluation: lose the value path's
  dynamic range. Direct products remain an independent high-precision oracle.
- Accepting coincident-center or point-collision derivatives merely because the
  value-only grid has a prescription: those branches require separate validation.
- Advertising complete DFT forces: grid motion is only one missing source.

## Invariants

The old value-only quadrature, ordering, radius rules and coincidence behavior
remain unchanged. The derivative interface rejects its unvalidated coincidence
and collision domain, keeps point/weight sources separate, and never grants SCF
state validity. Source/direction identities and immutable outputs prevent silent
reuse across a different geometry/displacement. Pair branch identities are not a
substitute for checking finite-displacement topology changes.

## Evidence

Small CPU tests compare independent value-only multistep differences and a
60-digit Decimal direct-product oracle. Rebuilt atom-centered grids test owner
motion and physical weights. A scalar field contracts through existing
GeometryPartials; omitting either point or weight response fails a dedicated
gate. Translation, atom permutation, exact-zero factors, partial tiles and
runtime-free source generation are covered without native execution or GPU use.

## Consequences and revisit conditions

This is a compiler-side CPU building block with O(tile_points * atom_count)
numeric storage, not native resource-ledger integration or performance promotion.
B2 must bind these sources to the real stationary-state identity and assemble all
remaining gradient components; C owns bounded native lowering and hardware gates.
Expand coincidence/collision domains only with a precise branch derivative and
independent evidence. Do not add a permissive fallback that silently drops terms.

## References

- Issue #163 B1; merged #455 supplies A.
- `python/vibeqc_compiler/xc/grid_response.py`
- `tests/python/test_grid_response.py`
- `docs/dft_grid.md`

Agent: ChatGPT
Model: GPT-6 Astra Pro
