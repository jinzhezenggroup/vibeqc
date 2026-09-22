# Decision: make exact CC state transport a complete-identity operation

Status: implemented
Date: 2026-09-22

## Problem

An RCCSD amplitude array does not identify its reference determinant, orbital
frame, Hamiltonian or equations. Shape-only reuse can silently cross an
occupied-virtual rotation, changed frozen-core definition or a different
integral/equation approximation. The existing `AmplitudeSnapshot` correctly
allowed only the identical reference ID, but provided no diagnosed path for an
exact change of orbital coordinates and no separate classification for future
cross-basis projection.

## Decision

State transport is a standalone CPU/reference layer in `tools.vibeqc_cc`, not a
controller branch. A request binds source and target reference, geometry, basis,
occupation, core, Hamiltonian, equation, provider and approximation identities
to coefficient-frame hashes, the actual coefficient arrays and target/source AO
overlap. Reference precision, screening/overlap policies, energies and
functional/grid identities are explicit rather than hidden behind unequal
opaque snapshot IDs.

The layer distinguishes identity, exact block-unitary orbital rotation,
projected warm-start candidate and incompatible reference change. Only identity
and exact rotation may emit target-bound amplitudes. Basis changes that preserve
occupied rank but are not isometries remain labeled projection candidates and
raise if exact rotation is requested. Arbitrary occupied-virtual mixing, lost
occupied rank and all unsupported identity changes fail closed.
Exact block maps must additionally preserve the occupied/virtual orbital-energy
operators and reference energy. Nonempty frozen-core transport is rejected;
future support must distinguish the physical core and active-occupied subspaces.
The diagnosed decision is factory-only so arbitrary maps cannot be relabeled as
exact after classification.

## Rejected alternatives

- Matching T1/T2 shapes was rejected because it cannot distinguish a changed
  determinant, core mask or scientific operator.
- Energy-order orbital matching was rejected because phases, permutations and
  rotations within degenerate subspaces are physically arbitrary.
- Automatically projecting missing virtual components was rejected from slice A
  because target denominators, missing-component policy and target residual
  refinement belong to the later projected-warm-start slice.
- Embedding this decision in a TargetProblem/StagePlan controller was rejected;
  that ownership belongs to issue #192.

## Invariants

- Maps are metric cross overlaps, `C_t^T S_ts C_s`, with explicit occupied and
  virtual partitions.
- Exact T1/T2 transforms apply the target-from-source occupied/virtual map to
  every amplitude axis and preserve restricted simultaneous-pair symmetry.
- Equal dimensions, source convergence or a transported energy never establish
  target compatibility or target convergence.
- Projected classifications cannot create exact amplitudes; target residual and
  energy recomputation remain mandatory before any target result is accepted.
- Solver histories are not transported by this layer.
- Coefficient arrays must match their endpoint frame hash, and only the
  classifier may construct a transport decision.
- Current support is unscreened FP64 all-electron RHF. Equal frozen-core index
  tuples do not prove equal physical core subspaces.

## Evidence

`tests/python/test_state_transport.py` compares staged production transforms
with independent explicit loops and checks transformed one-/two-electron RCCSD
energy terms. It covers identity, phase/permutation and degenerate rotations,
rectangular basis expansion, virtual nullity, occupied-rank loss,
occupied-virtual mixing and reference/core/equation/provider/approximation
identity rejection. Regressions also cover screening/operator changes,
identity/array mismatches, forged exact decisions and core/active occupied
mixing under equal frame-local core indices.

## Consequences

The first implementation is intentionally a small CPU/reference transform. It
allocates complete target T1/T2 outputs and is not a bounded GPU transport
kernel. Callers gain a reusable compatibility proof without gaining projected
amplitudes, solver-history reuse or orchestration.

## Revisit when

Issue #190 slice B defines a validated projection/missing-component policy and
requires target CC residual refinement, or slice C introduces bounded native
GPU/response transport. A controller may consume this decision after #192 owns
the corresponding TargetProblem/StagePlan boundary; it must not duplicate the
taxonomy.

## References

- Issue #190
- Issues #147, #148 and #189
- `docs/state_transport.md`
