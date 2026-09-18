# Strict DF final-state contract

The internal `scf/solver/final_state.hpp` contract selects CUDA DF RHF/UHF
energy, force and physical-reference states independently of the eigenprovider.
The CUDA snapshot producer below retains candidates for this contract. The
public C/Python result layout is unchanged.

A candidate binds its prepared basis/geometry/representation/device owner,
source item, solve epoch, orbital and density generations, resolved J/K model
and occupied counts. The prepared owner must supply immutable unique identities;
dimensions cannot establish ownership. A new solve must advance its epoch even
when local iteration counters restart. A producer retains both the candidate's
output-density generation and the generation at which its origin Fock was
evaluated. Zero identities and effective/DIIS/shifted origins fail validation.

Selection always evaluates a physical Fock at the supplied density through a
borrowed synchronous callback. The provider must evaluate that density for its
bound model and return the exact current identity; it cannot relabel a cached
Fock. Origin Fock generations may precede the candidate density only when the
frame passes against this current evaluation. This numerical test does not
require artificial equality of adjacent nonlinear SCF iterates.

Validation checks finite symmetric matrices, ordered eigenvalues, physical
eigen residual and S-orthogonality, reconstruction of D from occupied C, the
commutator `F D S - S D F`, electron counts and `D S D = occupation_weight * D`.
RHF uses weight two; UHF uses one for each spin, including an empty beta block.
The existing absolute eigenframe gates are unchanged. Density RMS and maximum
commutator are capped at the smaller of the requested density tolerance and
1e-8; trace and metric idempotency errors are capped at 1e-8. Energy is recomputed
as the nuclear term plus `1/2 sum_spin Tr[D (H + F)]` for the resolved quadratic
J/K model. The contract does not implement a KS/XC energy functional.

Force selection additionally solves the current physical Fock and compares its
occupied projector with the candidate density. Both maximum elementwise and RMS
fixed-point defects must satisfy the smaller of the requested density tolerance
and 1e-8. Reconstructing the retained frame is a separate check: zero
reconstruction error does not prove that its density is a fixed point of the
current physical Fock. Energy-only selection does not perform this extra probe.

A rejected or absent candidate enters bounded strict correction. Each correction
diagonalizes the current physical Fock through the borrowed operation, validates
its output before projection, advances both determinant generations, and
evaluates the physical Fock again at the new density. Success requires all state
checks and the requested energy-change tolerance. A single rebuild is therefore
insufficient when its energy change or current-F residual fails. The default
budget is four corrections; zero allows validation only, and values above 64
are invalid. Generation overflow, exhaustion and provider failures return no
successful state. An independent later call can recover without retained state.
A rejected fixed-point probe supplies its already checked eigenframe and
projector to the correction, avoiding a duplicate solve at the same density.

The explicit force-rebuild control executes that same solve/project/evaluate
path even for a valid candidate. Counts report actual physical evaluations,
per-spin eigensolves, joint density updates and rejected candidates. The host
component ledger separately records validation, physical Fock construction,
strict correction and weighted-density construction. It uses the existing
`final_fock` reason for actual provider solves. Fixed-point checks and promotions
are also explicit: per-spin solves equal corrections plus checks minus promoted
checks. Successful force items each contribute one accepted check; corrections
remain bounded by the original limit.

Energy-only selection leaves W empty. Requests needing W compute
`D F[D] D / occupation_weight` only after the state passes; there is no cached-W
input. This equals the orbital-energy expression at an exact stationary state
while using the current physical Fock when a retained frame has lagged orbital
energies. A downstream occupied factor must match the verified determinant
identity and density witness. Downstream integration must additionally enforce
the solve epoch and full model identity, budget snapshots and scratch, preserve
all DF derivative terms, and validate physical-reference export.

`tests/native/test_final_state.cpp` supplies analytic nonidentity-metric RHF/UHF
frames, degenerate rotations, independent energy/W expectations, identity and
numerical faults, actual rebuild counts, old-factor rejection and an oscillating
nonlinear Fock. Small-gap fixtures distinguish reconstruction from a physical
fixed point and verify promotion without duplicate solves. Neither production nor reference diagonalization constructs the
expected answers. Molecular forces, device snapshot lifecycle and performance
claims require the later integrated paths and their independent qualification.

## CUDA candidate snapshots

`cuda_density_fitting_final_state.hpp` exposes a version-one eligibility token
and detached snapshot reader. The existing persistent SCF owner now stores full
C/epsilon separately for each spin and item, together with the producing solver
status and output-density generation. The copy runs under the old active mask,
before the density commit can deactivate an item. UHF alpha is saved before
beta overwrites shared coefficient scratch. Ordinary launches and captured
graphs execute the same masked store; graph construction counts no real copy.

Every attempted device solve invalidates the previous eligibility and advances
a plan-owned epoch before input checks. The epoch survives persistent-storage
rebuilds and fails closed at saturation. Only converged successful items are
published after the final density readback. Imported D has generation one;
after `iterations` committed projections, the candidate density has generation
`iterations + 1` and its physical-origin Fock was evaluated at generation
`iterations`. This explicit offset is separate from the occupied-K kernel's
local iteration counter.

The token includes the immutable source owner/item, solve epoch, determinant
generations, the compact loop's canonical FP64 full-range DF HF model and spin
occupations. Snapshot reads require its exact current identity before transfers,
then check retained device generation, solver status and finite data. The reader
copies only that item's C/epsilon and actual retained D, converts coefficient
columns to the row-major host convention and drains the plan stream before
releasing any destination. It rejects capture and never treats these candidates
as a verified final physical Fock. All operations borrow a serialized plan owner.

Both native planners and the composed Python resource plan charge the two-spin
upper bound `batch * [16 * (n*n + n) + 24]` device bytes. Actual RHF allocation
uses one spin; UHF uses both, including empty beta. This capacity is per item,
and the independent cold retry receives a complete single-item share in addition
to its own ordinary solver allowance. Host eligibility metadata is also reserved.
No unbounded history or force-only W is retained by the snapshot owner.

`test_df_final_snapshot.cpp` exercises analytic rotated RHF/UHF frames in batches
one/four, early inactive neighbors, scratch poisoning, distinct alpha/beta
storage, stale owner/model/epoch/generation/occupation tokens, corrupt device
generation/info, invalid and nonconverged replay, epoch exhaustion and recovery.
Physical candidate checks use the independent contract and analytic F/S/D.
Production consumers validate these candidates as described below; performance
acceptance remains a separate measurement.


## CUDA energy, complete forces and reference selection

CUDA RHF/UHF requests read the explicitly successful device candidate, verify
its exact returned-D witness, requested occupations and metric policy, evaluate current physical J/K,
and run strict selection. A qualifying energy-only state uses one physical Fock
evaluation and no final solve. A force state also performs a physical fixed-point
probe, using the ordinary provider and counting its Fock download and eigensolve.
Necessary corrections use the qualified ordinary provider,
project and re-evaluate D, and must pass all invariants and the energy-change
gate within sixteen corrections. The larger consumer budget covers the strict
current-F convergence observed after cold and changed-geometry SCF; no numerical
gate is relaxed. Provider failures propagate without a hidden
host J/K substitution. Failed strict selection clears the convergence claim.

Host DIIS recovery supplies no compact candidate. Its current solve epoch and
an offset generation range distinguish recovered D from compact iterations
without invalidating another item's token. It therefore enters real correction.
Both UHF spin channels are selected jointly. Each correction evaluates one pair
of physical Fock matrices and uses two provider solves, including the empty
beta channel; a rejected fixed-point probe can supply those solves. Complete
forces consume the selected density and its matching W
exactly once in the existing one-electron/Pulay, raw three-center, auxiliary
metric and nuclear response. Energy-only selection constructs no W and invokes
no derivative consumer. CPU finalization preserves its independent sequence.

Physical-reference export additionally requires maximum reconstructed-D drift
and absolute `C^T F C - diag(epsilon)` error at most 1e-8. These optional gates
are part of selection, so their failure enters correction. The accepted full
C/epsilon is moved into the exported reference, which retains its independent
completed-reference validation; no extra export eigensolve is needed. The
analytic contract tests separately exercise amplification by an ill-conditioned
metric and a maximum-D error hidden by RMS.

`VIBEQC_DF_FORCE_FINAL_REBUILD=1` forces actual solve/project/evaluate work even
when a candidate qualifies. `VIBEQC_DF_REFERENCE_FINAL_EIGEN=1` additionally
selects the independent reference provider and also forces that work. Existing
setup/final provider ablations explicitly force final rebuilding on both sides
so their execution remains provider substitution. Normal preparation ablations
retain the normal selection path. The host ledger records every actual physical
Fock, validation and strict correction, plus the accepted reuse/corrected outcome.

The composed resource plan reserves `8 * (32*n*n + 8*n)` additional host bytes
for one serialized item's detached candidates, verified output and bounded
validation/correction products. The existing per-item device snapshots and
independent ordinary correction/retry workspace remain separately charged.

The matrix allowance covers both spin channels, canonicality products and W.
Candidate, corrected, probed and accepted epsilon vectors can coexist; the
eight-vector allowance covers both spins of all four frames. The
selection is serialized across items; no per-iteration history is retained.

Molecular qualification exercises Cartesian/spherical RHF water and UHF OH,
batches one/four, resident/constrained budgets, cold/warm/changed geometry,
frozen seeds, output transitions and failed neighbors. Independent CPU total
DF energy differences and translation checks exercise the full force sum,
including H2+ with empty beta. Polarized H2 reference export compares D, canonical
energies and full forces against CPU and separates cold/retained/device-rebuild/
reference-rebuild traces. Timing claims require separate clean endpoint evidence.

See the [numerical-state decision](../.agents/notes/implemented/numerics/2026-09-17-full-rank-response.md)
for the independent fixed-density evidence and retained failed candidates.
