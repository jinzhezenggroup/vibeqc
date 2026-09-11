# Incremental Coulomb factorization

The experimental post-HF provider starts from the existing unscreened native
raw integral source. `CoulombColumns` reads bounded diagonal/column pieces;
`IncrementalCholesky` retains a fixed-capacity factor prefix and appends actual
new pivots. It never constructs a molecular four-index tensor. This initial
implementation supplies CPU factorization and J/K/MO consumers, a native CUDA
execution policy, and staged RHF initialization with exact-target cleanup.
These remain experimental internal post-HF tools, not installed public APIs.

`PairSpace` enumerates `(mu, nu)` with `nu <= mu` in row order. Its normalized
coordinate is `sqrt(m) * D[mu,nu]`, where multiplicity m is one on the diagonal
and two off diagonal. The factored matrix is
`M[i,j] = sqrt(m[i]*m[j]) * (mu nu|rho sigma)`. Thus `L.T @ L` approximates M
and ordinary packed dot products preserve the full matrix Frobenius product.
Recover a symmetric physical factor with `B[mu,nu] = L[P,i] / sqrt(m[i])`.
`PairSpace.pack` rejects asymmetric inputs instead of silently projecting them.

`refine(threshold, maximum_rank=...)` uses an absolute pair-matrix diagonal
threshold. Exactly tied maxima choose the smallest pair index. Calls preserve
all completed columns and append pivots up to the reserved rank capacity.
Each pivot validates its entire Schur column before commitment. A later source
failure leaves the valid completed prefix visible; it does not undo earlier
pivots. The factor identity hashes actual factor bytes, source identity, pair
convention and precision. A rank change invalidates captured consumer and solver
history, including DIIS/Krylov history, and requires fresh residuals.

Finite precision can produce small negative diagonal residuals. Values larger
than `64 * eps * max(initial_diagonal) * (rank + 1)` in magnitude fail; smaller
negative values are zeroed, with their count and largest magnitude recorded
for each pivot.
Column Cauchy checks test additional necessary PSD conditions. A requested
threshold below this rounding floor can return `roundoff_limited`, explicitly
distinct from `threshold_met`. `rank_limited` means the requested rank cap was
reached first. Tightening a threshold never relabels an approximate Hamiltonian
as exact conventional HF or CC.

The maximum diagonal bounds individual residual entries and the residual trace
bounds spectral/Frobenius norms **assuming a PSD Schur complement in exact
arithmetic**. The reported finite-precision quantities are tensor diagnostics,
not certified observable errors or proofs that an arbitrary supplied matrix is
PSD. They do not certify relaxed energy or force accuracy. Rank/pivot derivatives
are unsupported; attaching exact-integral gradients to a truncated-factor energy
would not define consistent forces.

Shared `ResourceBudget` planning precedes source reads and factor allocation.
It counts the existing source and recurrence allowance, reserved factor
capacity, diagonal/column/update workspace, raw tiles and detached factor-tile
publication. Factor exports cannot exceed `export_rank_tile`; retained copies
belong to the caller after return. Python object metadata and BLAS runtime
overhead are explicit exclusions. No new user-level memory cap or cache is
introduced.

Tests compare incremental and from-scratch factors, deterministic pivots,
packed/full contractions, partial tiles, zero/tiny/ill-conditioned matrices,
failed updates, immutable exports and two memory limits. Independent pinned
PySCF AO fixtures validate raw columns and tight-threshold recovery for H2,
water and LiH, plus selected spherical f-shell columns.

## Fixed-generation consumers and accuracy

`LowRankProvider(factor, snapshot=None, budget=...)` captures one immutable
factor identity. `jk(D)` returns raw J[D] and K[D] for one real symmetric
density. Passing alpha/beta densities returns J[Da+Db] and separate Ka/Kb.
Occupation and exchange prefactors belong to the consuming RHF/UHF method.
The implementation streams physical factors and uses matrix contractions;
there is no global four-index AO tensor or retained rank-by-AO-by-AO array.

With a validated exact `ReferenceSnapshot`, `get(MOBlock(...))` produces only
the requested chemists' MO block. Source and reference must agree on geometry,
basis, representation and electron count. The snapshot's closed-shell 2/0
occupations require source multiplicity one. Its `reference_id` continues to identify
the exact orbitals, while `hamiltonian_id` identifies the Cholesky correlation
approximation. Existing same-Hamiltonian MP2/CC entrypoints reject this mixed
reference/correlation combination; an approximate-correlation method adapter
must define it explicitly. No reference energy or orbital identity is relabeled.
Refining the factor invalidates an existing consumer before J/K or MO output;
create a new view and rebuild dependent residuals/history.

`audit_fixed_density` compares the RHF two-electron energy with the original
unscreened raw provider at identical density. It returns the existing NUM01
`ErrorEvidence` with source `integral_factorization`, scope `fixed_density`,
and distinct target/evaluated model identities. The shared accuracy assessment
therefore cannot treat it as a successful relaxed-target calculation. Both the
observed difference and conditional tensor diagnostics remain available for
later refinement decisions. Consumer/audit memory is composed with the same
factor resource owner; impossible combined allocations fail before execution.

## CUDA execution and staged initialization

`CudaIncrementalCholesky` keeps actual factor prefixes on both host and device.
CPU native raw integrals supply one column at a time; CUDA subtracts the
retained prefix in deterministic rank order. CPU owns pivot selection,
PSD/roundoff checks and the final commitment decision. Only the raw, projected
and newly committed columns cross the device boundary during refinement.
The host mirror supplies bounded CPU MO transformations. Native J/K streams
one physical factor through cuBLAS contractions; outputs are explicitly copied
to host. No new shell/operator formula or separate CUDA runtime is introduced.

Shared resource preflight counts both factor mirrors, numeric workspace and
the reused CUDA Context's cuBLAS allowance before source reads/device creation.
The native arena is checked against the plan, and execution performs no device
allocation. A failed factor upload invalidates the device owner; later calls
fail closed. CUDA section timing events are enabled and can add substantial
overhead for tiny contractions. Reported section times exclude some host and
runtime work; whole-solve wall time is the comparison metric.

`solve_refined_rhf(factor, target, stages)` consumes explicit `RefinementStage`
objects with monotonically tightening pair-diagonal thresholds/rank caps and
bounded iteration counts. A simple damped host RHF initializer carries only
the density between stages. Each changed factor generation gets a new consumer
and a fresh Fock/commutator residual. No DIIS/Krylov history is retained. The
electronic-energy jump at the same density records observed sensitivity to
refinement; it is not a bound on relaxed observables. This predetermined policy
does not yet choose thresholds automatically from a requested energy/force error.

Final cleanup always calls the existing `FockPlan.solve(initial_density=...)`
for the identical unscreened full-Coulomb closed-shell RHF target. That solve
starts fresh DIIS history and must converge; failure propagates. Only its
energy, density and optional exact forces appear in `result.exact`. The wrapper
rejects DF/range/operator, geometry/basis and ensemble mismatches. Factor/pivot
derivatives remain unsupported, and no approximate-stage energy is labeled
consistent with the exact forces. The initializer's shared memory plan covers
its factor/source and numeric workspace; the separately owned exact FockPlan
retains its own native resource controls, reported separately.

`tools/validate_low_rank.py` compares fixed-rank and staged initialization plus
exact cleanup with direct exact SCF. Existing fixed-auxiliary DF is measured
separately, with explicit energy/force differences from the exact target.
Independent pinned H2/water/LiH fixtures gate final energies/densities; matched
exact cleanup gates forces. Small dense reconstructions gate same-approximation
J/K and conditional pair residual diagnostics only in the reference driver.
Raw timing samples include setup, source reads, rank extension, transfers and
cleanup through forces. Compilation, destruction and reference comparisons are
outside that clock. The small fixtures establish correctness and measured
overhead, not a production speedup or large-system scaling claim.
