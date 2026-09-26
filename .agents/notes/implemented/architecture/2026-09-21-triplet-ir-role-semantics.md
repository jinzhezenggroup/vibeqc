# Decision: keep TripletIR role ordering explicit

Status: implemented
Date: 2026-09-21

## Problem

Three-body angular science needs deterministic topology identity without pretending
that every permutation of three atoms is scientifically interchangeable.  The
first planned consumer, GFN1 halogen correction, has donor/neighbor/acceptor roles,
so treating a triplet as an unordered atom set would erase information before the
scientific expression is lowered.

## Decision

`TripletTopology` stores canonical **role-ordered** `(i, j, k)` entries.  The
second atom `j` is the angle center and deterministic owner.  Entry order is
lexicographically canonical and exact duplicate entries are forbidden, while
swapping `i` and `k` deliberately changes topology identity.  The generic
lowering exposes both center-relative vectors, distances, their dot product and
cosine through shared TensorIR.  System reduction and coordinate JVP/VJP are also
shared TensorIR operations.

Runtime neighbor/topology rebuild policy remains outside the compiler contract.
Scientific consumers decide which role-ordered triplets exist and bind their own
parameter identity.

## Rejected alternatives

An unordered `i < j < k` representation was rejected because it cannot preserve
role semantics for directional three-body corrections.  A method-specific GFN1
triplet kernel was rejected because it would duplicate geometry algebra and AD
between CPU and CUDA paths.

## Invariants

- all three atom indices are distinct and in range;
- the exact role-ordered triplet list, convention, ownership and scientific
  parameter identity participate in compiler identity;
- changing roles changes topology identity even when a symmetric qualification
  expression happens to produce the same numerical angle;
- primal and coordinate VJP lower through the same TensorIR graph;
- public method admission and topology rebuild scheduling stay runtime-owned.

## Evidence

`tests/python/test_geometry_triplet_ir.py` covers malformed and duplicate
triplets, role-sensitive identity, angle geometry, deterministic system reduction,
translation/rotation invariance, finite-difference coordinate VJP, force-sum
translation conservation, stale execution identity, and CUDA lowering of primal
and generated reverse programs.

## Consequences

Future three-body consumers can share the geometry/AD/lowering contract while
retaining their own scientific topology rules.  A consumer whose terminal roles
are provably symmetric may canonicalize those roles before constructing the
compiler topology, but the generic compiler does not assume that symmetry.

## Revisit when

A production consumer requires a genuinely unordered or permutation-symmetric
many-body topology that cannot be represented by role normalization at its
frontend boundary.

## References

- #853
- #837
- #852
