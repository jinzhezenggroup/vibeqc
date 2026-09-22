# Decision: Preserve pinned GFN1 halogen topology as role-ordered TripletIR

Status: implemented
Date: 2026-09-22

## Problem

The GFN1 halogen correction is a genuine three-atom angular term. Encoding it
as pairs would lose the donor's nearest-neighbor role and make Cartesian
derivatives depend on a second, handwritten scientific implementation. The
pinned tblite source also has discrete cutoff, nearest-neighbor and tie-break
semantics that must be part of replay/cache validity.

## Decision

Use role-ordered `(nearest-neighbor, halogen-donor, acceptor)` triplets, with
the donor as the TripletIR center. Build donor/acceptor entries at the inclusive
20-bohr cutoff, select the closest positive-distance neighbor with a strict
comparison, and therefore retain the lowest atom index on exact ties. Express
the angular and radial energy in TensorIR and obtain the Cartesian VJP from the
same graph. Bind the resulting program to the canonical GFN1 XtbMethodIR
`halogen_correction` primitive without changing public runtime capability.
The pure equation/topology stays under `geometry`; the MethodIR-bound wrapper
lives under `method` and accepts the actual `XtbMethodIR` type. The compiler
dependency guard explicitly permits method composition to consume geometry,
while geometry remains independent of method policy.

When the selected neighbor is the acceptor, the pinned angular factor and its
first derivative are identically zero. Omit that entry because generic
TripletIR deliberately rejects repeated atom roles. Reject coincident
donor/acceptor coordinates because the pinned radial expression divides by
their distance.

## Rejected alternatives

- A fake pair term cannot represent or invalidate the selected neighbor.
- Copying tblite's handwritten gradient would create a second derivative
  implementation and would not qualify compiler AD.
- Allowing repeated indices in generic TripletIR would weaken the Stage-A
  three-distinct-role invariant for a term that is provably zero.
- Duck-typing or copying the GFN1 parameter-set hash into geometry would allow
  structural impostors to claim canonical MethodIR provenance.
- Smoothing the cutoff or nearest-neighbor selection would change the pinned
  GFN1 model rather than merely lower it.

## Invariants

- Donors are Cl/Br/I/At and acceptors are N/O/P/S.
- Topology changes at the inclusive 20-bohr boundary, a nearest-neighbor switch
  or an exact-tie ordering change require recompilation/rebuild.
- Cartesian derivatives are valid only within a fixed topology region; no
  derivative is claimed at a discrete boundary.
- Parameter identity binds the canonical GFN1 JSON plus tblite revision
  `fa8a4416e8fe093d0075bc10ac875494c2a449a9` and halogen source SHA-256
  `ed3469a1e07d95d75bb09b1a4615616a9416aff425c9cc46ae057dca450ad449`.
- Method-bound identity also binds the exact GFN1 MethodIR primitive.
- Method-bound JVP/VJP access requires the `nuclear-gradient` compiler product;
  the lower method-neutral geometry program remains differentiable for
  qualification.

## Evidence

- Pinned tblite goldens: Br2-NH3, Br2-OCH2 and FI-NCH.
- Independent scalar-oracle comparison and centered finite differences at
  multiple step sizes.
- Role, duplicate, cutoff, coincidence, topology invalidation and cache
  identity tests.
- Translation, rotation and atom-permutation covariance tests.
- Shared TensorIR CPU execution, generated reverse AD, CUDA source lowering
  and explicit real-device energy/VJP test coverage.

## Consequences

Runtime code only schedules topology rebuilds and execution resources. The
compiler owns the scientific expression, exact discrete topology contract and
derivative generation. The overall GFN1 method remains non-executable until
the separate SCC/eigensolver/public-admission work is complete.

## Revisit when

The pinned upstream GFN1 halogen model changes, periodic translations are
admitted, or a replacement topology rule is independently qualified.

## References

- Issue #853 and parent #837.
- PR #857 (generic TripletIR Stage A).
- `docs/compiler_gfn1_geometry.md`.
- `upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3/gfn1_manifest.json`.
