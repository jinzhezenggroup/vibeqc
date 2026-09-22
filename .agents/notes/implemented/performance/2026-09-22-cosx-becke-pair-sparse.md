# Decision: Keep Becke pair derivatives sparse

Status: implemented
Date: 2026-09-22

## Problem

`MolecularGrid::contract_weight_derivative` evaluates every atom pair for every
materialized grid point.  A pair derivative can affect only the grid-point
owner and the two atoms in that pair, but the original implementation stored it
in a `3 * atom_count` vector.  It cleared, scaled, and contracted that dense
vector inside the pair loop.  For a fixed per-atom grid, this added a full atom
coordinate sweep to every pair and raised the complete weight-response work
from cubic to quartic in the atom count.

## Decision

Represent each pair derivative as at most three three-coordinate blocks.  The
owner, first pair atom, and second pair atom are coalesced before evaluation so
an owner that is also a pair atom occupies one block.  Becke polynomial slopes
and partition-log updates visit only those blocks.  The dense partition
normalization and final molecular-coordinate contraction remain unchanged.

## Rejected alternatives

- Retaining the dense temporary preserves simple indexing but repeats work on
  coordinates that are known to be zero.
- A general sparse container adds allocation and lookup overhead to a hot loop
  whose support has a fixed upper bound of three atoms.
- Moving the analytic Becke response to CUDA would change ownership and
  numerical acceptance boundaries beyond the fixed-density COSX force scope.

## Invariants

- Contributions are accumulated in owner, first-atom, second-atom order before
  applying the partition slopes, matching the dense implementation when atom
  identities overlap.
- Coincident-center tolerance, `mu` clamping, partition iterations, logarithmic
  normalization, and final accumulation semantics are unchanged.
- CPU reference and CUDA molecular derivative paths continue to share this
  exact host Becke-weight response.

## Evidence

- `tests/python/test_cosx_derivative_resource_contract.py` contains a
  failure-first source contract that rejects a nuclear-coordinate sweep inside
  the pair contraction.
- `vibeqc_cosx_reference_tests` checks the analytic molecular derivative against
  finite differences and its translation/permutation properties.
- `vibeqc_cosx_cuda_tests` checks CUDA/CPU molecular-gradient parity, changed
  geometry, multiple tile sizes, translation, and higher-angular-momentum
  coverage.
- `vibeqc_cosx_scf_tests` checks RHF/UHF stationary forces and multi-step SCF
  finite differences.

## Consequences

The pair-processing portion now has constant coordinate work per pair.  With a
fixed number of grid points per atom, the complete host weight response is
cubic rather than quartic in atom count.  The existing dense per-atom partition
Jacobian remains the dominant storage and cubic-work term.

## Revisit when

Revisit this representation if the partition function gains support on more
than the owner and pair atoms, or if the complete analytic weight response moves
to a separately qualified device implementation.

## References

- Issue #246
- PR #1007
- Review thread `discussion_r4068912306`
