# Issue 162 KS final-state handoff design

## Scope

Close the remaining Issue #162 acceptance gap by exposing an internal,
immutable snapshot of a successful native KS state for later use by Issue
#163. This change does not implement nuclear gradients, enable force
capabilities, add density-fitted DFT, or change SCF convergence equations.

## Architecture

Add a KS-specific snapshot boundary beside `CudaKsPlan`. The resident plan
remains the owner of device allocations and publishes a versioned eligibility
token only after a successful converged solve. A detached snapshot reader
validates the exact current token before copying state needed by a downstream
stationary-gradient consumer.

Reuse the existing `scf::solver::FinalStateIdentity` and density-factor
generation vocabulary where their semantics match. Add KS model identity for
the data not represented by the HF final-state contract: GridSpec, functional
and regularization policy, resolved Coulomb provider, spin mode, prepared
source owner, and plan/device ownership. Dimensions alone never establish a
match.

## Published state

The snapshot contains one or two spin blocks of:

- converged density `D`;
- matching physical non-DIIS Fock `F`;
- canonical coefficients `C` and orbital energies `epsilon`;
- occupied counts and explicit occupation weights;
- energy-weighted density `W = C diag(f * epsilon) C^T` when requested;
- the physical residual and energy components associated with the same state;
- immutable geometry, basis, grid, functional, provider, solve-epoch, orbital,
  Fock, and density generation identities.

The normal energy-only result does not construct or transfer `W`. The snapshot
reader constructs it only for an explicit internal downstream request.

## State consistency

The resident CUDA iteration already evaluates the physical Fock at the current
density and diagonalizes that Fock to form the proposal. On convergence, the
plan must retain the coefficient/eigenvalue frame that produced the accepted
proposal and bind it to the evaluated density/Fock generations. Publication
requires finite data, successful eigensolver status, physical convergence,
electron/spin traces, Fock-density consistency, S-orthogonality, eigen
residuals, and reconstruction of the accepted density within the existing
gates.

If the resident trajectory's accepted density and retained orbital frame are
not sufficiently consistent, snapshot publication fails closed. This Issue
#162 slice will not add an independent correction solver or silently publish a
nearby state; Issue #163 may later consume the common strict-selection
machinery if correction is required.

## Invalidation and lifecycle

Every attempted solve advances a nonzero plan-owned epoch and invalidates the
prior eligibility token before input validation. A new geometry, basis, grid,
functional, spin state, provider strategy, precision policy, or prepared owner
requires a new identity. Failed, nonconverged, pending, or replaced solves
cannot publish a snapshot. Warm density reuse does not reuse the previous
state identity.

Snapshot reads borrow the serialized plan owner, verify token equality before
and after device work, and synchronize before releasing destination storage.
No global cache or unbounded iteration history is introduced.

## Interfaces

Introduce an internal header with:

- a versioned KS final-state token;
- a detached KS final-state snapshot;
- a query that returns eligibility for the current successful plan state;
- a reader that validates an expected token and optionally constructs `W`.

The public C and Python calculation result layouts remain unchanged. Existing
energy-only and unsupported-force behavior remains unchanged.

## Diagnostics and resources

Extend internal diagnostics with solve epoch, determinant/density generations,
snapshot eligibility, and actual snapshot transfer/synchronization counts.
Charge retained coefficients, orbital energies, physical Fock, generation
metadata, and optional detached host state through the existing KS resource
plan. Do not count an estimate as an executed transfer.

## Tests

Add native contract tests covering:

- RKS and UKS, including an empty spin channel;
- exact `D/F/C/epsilon/occupations/W` reconstruction and consistency;
- batch-neighbor isolation;
- fixed-geometry warm replay with a new solve epoch;
- changed geometry, grid, functional, provider, spin, and policy rejection;
- stale owner/epoch/orbital/Fock/density generation rejection;
- pending, failed, nonconverged, eigensolver-failed, and nonfinite states;
- explicit no-`W` energy snapshots and requested-`W` snapshots;
- transfer/resource counts matching actual operations.

Run CPU/no-device structural tests locally. Build and execute the allocated-GPU
tests on the project validation environment, recording skipped and not-run
cases separately.

## Completion boundary

Issue #162 is ready to close when the tests demonstrate that a successful
native KS state exposes or reconstructs the complete current
`D/F/C/epsilon/occupations/W/grid/provider` identity required by Issue #163,
and the evidence documents the remaining transfer/component boundary. Full
stationary LDA/GGA force assembly remains exclusively in Issue #163.
