# Shared orbital response and bounded Krylov solves

`tools/vibeqc_response` is the shared native response-solver layer for RHF and
semilocal-KS orbital response. It separates the problem snapshot, the
matrix-free operator, and the linear-solver/recycling state so downstream
property, Hessian, and correlated-gradient code can reuse one implementation.
This slice is partial: UHF response and the native converged RKS/UKS CPKS
endpoint remain open acceptance items for `#179`/`#162`.

This layer is not a new public electronic-structure method. RHF remains the
registered HF method, and `#162` still owns the converged RKS/UKS SCF endpoint.

## Problem snapshot

`ResponseProblem` binds all scientific state before an operator or subspace is
created:

- immutable converged reference arrays and reference identity;
- explicit occupied and virtual spaces, restricted spin block, and the
  occupied-major/virtual-minor nonredundant rotation layout;
- overlap metric, canonical gauge, RHS layout, operator identity, and
  Hamiltonian/functional/grid model hash.

Changing a reference, even with the same dimensions, changes
`reference.identity` and therefore `ResponseProblem.identity`. A retained
Krylov space checks this compatibility identity before every use. Reusing
vectors across a geometry/basis/model change requires an explicit
`KrylovRecycleSpace.transport(..., transform)` call and re-evaluation; equal
vector length is never treated as compatibility.

`RotationLayout` stores a response vector `x[i,a]` with
`x[i,a] = x_(occ_i, virt_a)`. Its density response is
`Delta P = 2 sym_ov(x)`, and its skew orbital generator is
`K[virt,occ] = -x` and `K[occ,virt] = x`. Occupied-occupied and
virtual-virtual rotations are not independent unknowns.

`ResponseProblem.diagnostics` gates on `minimum_ov_gap`, the smallest absolute
occupied-virtual orbital-energy denominator of the active rotations. A
same-occupancy (occupied-occupied or virtual-virtual) degeneracy is a redundant
direction and does not trip `require_stable`; a symmetry-degenerate occupied or
virtual subspace with a finite occupied-virtual gap remains solvable.

## RHF operator

`RHFResponseOperator` applies the closed-shell RHF Jacobian without assembling
an AO N^4 tensor:

```text
Delta P_mo = 2 sym_ov(x)
Delta P_ao = C Delta P_mo C^T
Delta F_ao = J[Delta P_ao] - 1/2 K[Delta P_ao]
A x       = (eps_virt - eps_occ) x + C^T Delta F_ao C
```

`NativeJKBackend` streams the existing native shell-tile ERI source and forms
only the O(N^2) Coulomb and exchange responses. `DenseAOResponseBackend` is a
tiny independent oracle used only by tests. The CUDA DF backend validates its
metric against the source auxiliary basis, geometry, and execution threshold;
a caller-supplied Hamiltonian label must match that metric. Supplying the
exporter's `metric` avoids rebuilding it during backend preparation.
`RHFResponseOperator.apply` is the JVP, `apply_transpose` is the VJP entry point, and `dot_identity` checks the
Euclidean transpose identity.

`explicit_rhf_response_matrix` independently assembles the tiny MO matrix

```text
A[(i,a),(j,b)] = (eps_a - eps_i) delta_ij delta_ab
                + 4(ai|bj) - (ab|ij) - (aj|ib)
```

and `finite_rotation_jvp` checks the same action against an explicit
`C exp(-t K)` rotation and an independent Fock build.

## Solver and multi-RHS strategies

`solve` implements restarted GMRES with a true residual at every configured
checkpoint. It reports the actual residual, iteration count, operator actions,
orthogonalization/operator timings, workspace bytes, and a non-success reason.
It does not silently regularize a singular denominator or claim success after
a workspace or stagnation failure.

The operator, preconditioner and retained-subspace roles are separate.
`DiagonalPreconditioner` is opt-in and rejects zero/near-zero diagonal entries;
it is never selected implicitly.

`solve_many` provides:

- `sequential`: one bounded solve per RHS;
- `blocked`: a shared orthonormal block basis with projected least squares;
- `recycled`: reference-bound retained vectors used as initial guesses and
  updated after each solve.

Rank-deficient RHS blocks are diagnosed. The workspace accounting includes the
retained Arnoldi/block basis and solver vectors. A small `max_workspace_bytes`
returns `workspace_limit` before applying the operator.

`SolveResult.relative_residual` is `||r|| / ||b||`; a zero RHS is defined as
`0` for an exactly zero residual and `inf` otherwise, never as an absolute
residual just because `||b|| < 1`. In the recycled strategy the budget sums the
shared immutable RHS copy, previously retained result arrays, independent
recycle-space vectors, projection and bounded replacement temporaries, and the
next solve workspace. The single-RHS API applies the same recycle reservation
before projection or operator application. A successful peak is a conservative
bound within the requested budget; a workspace rejection reports the required
bound. Operator/preconditioner storage and Python interpreter bookkeeping are
outside this solver numeric-buffer budget and require separate accounting.

## CPKS boundary

`FixedDensityXCDerivativeKernel` evaluates the audited semilocal feature
Hessian from #161 and contracts it with the exact first-order density-feature
response. `CPKSResponseOperator` adds that kernel to the J/K response action.
The kernel/reference basis, grid, functional, and density identities must
match exactly.

The [common contraction generator](xc_contractions.md) owns the scalar-Hessian
chain rule and AO assembly. `apply_spin()` preserves functional-spin channels
and cross-spin terms; `apply()` retains the restricted mean. An optional
matching `PreparedXCContractions` response owner selects bounded native CPU
execution through `prepared=...`, with its shared numeric resource plan.

Exact exchange, range-separated exchange, and unvalidated nonzero tau
derivatives fail closed. The public path also requires a converged `KS`
reference. Because the converged RKS/UKS endpoint is still owned by #162, no
native RHF reference is relabeled as KS and no CPKS endpoint is claimed from
an unconverged or mismatched reference.

## #153 interface

The correlated-gradient work in #153 should:

1. build its CC-specific orbital RHS and weights outside this package;
2. create one `ResponseProblem` from the exact converged RHF reference and the
   shared operator backend;
3. call `solve`/`solve_many` and require the returned true residual to meet its
   gradient gate;
4. retain only CC-specific RHS/weight state, not a second RHF CPHF/Z-vector
   implementation.

No SCF/DIIS iteration tape is part of this contract.

## Backend boundary

`NativeJKBackend` streams the CPU native shell-tile source. `CudaDFJKBackend`
owns a prepared streamed CUDA density-fitting J/K plan and applies the same
matrix-free RHF action on the RTX 5090 under Slurm. It fails closed when the
source has no auxiliary basis or the binary lacks CUDA support. The real-device
test compares this action with the independent explicit MO matrix from
`DFProvider`.

A direct four-center CUDA J/K response backend is not promoted by this slice;
the evidence labels that absence explicitly instead of treating the CUDA DF
path as equivalent to the conventional four-center Hamiltonian.

## UHF CPHF boundary

`UHFReferenceSnapshot`, `UHFSpinRotationLayout`, and `UHFResponseOperator`
provide the shared alpha/beta CPHF contract.  The packed vector stores all
alpha occupied-virtual rotations followed by beta rotations.  The matrix-free
action uses the native UHF Fock convention: Coulomb is evaluated from the
spin-summed density response, while each exchange action uses its own spin
density.  This keeps both spin channels coupled without storing an AO N^4
response tensor.

The UHF snapshot binds both canonical spin references, occupations,
Hamiltonian and SCF generation.  Consequently a recycle space cannot be
reused merely because alpha/beta dimensions happen to match.  The generic
GMRES, blocked multi-RHS and recycling APIs operate on this response problem
unchanged.

This is the HF UHF response layer only.  A native converged RKS/UKS CPKS
endpoint remains a dependency of `#162`; this module does not relabel a UHF
state as a KS endpoint or enable unsupported XC derivatives.

The direct CPU bridge can export a converged open-shell UHF solution through
`export_uhf`.  It canonicalizes the independently returned alpha and beta AO
densities, rechecks both physical commutators and density/Fock reconstruction,
and binds the result to the shared UHF response contract.  The bridge is
intentionally limited to the small direct CPU Hamiltonian: CUDA/DF UHF response
still fails closed until a spin-resolved device J/K response plan has separate
numerical and resource evidence.
