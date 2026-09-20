# Shared orbital response and bounded Krylov solves

`tools/vibeqc_response` is the shared response-solver tooling for RHF and
semilocal-KS orbital response. Its Krylov controller is Python/host-controlled;
the operator backends include native J/K execution. It separates the problem snapshot, the
matrix-free operator, and the linear-solver/recycling state so downstream
property, Hessian, and correlated-gradient code can reuse one implementation.
This slice is partial: the RHF response layer and the direct-CPU UHF response
layer (including `export_uhf`) and the native CPU LDA/PBE RKS/UKS CPKS handoffs are
delivered. Native CUDA CPKS and remaining performance acceptance stay open
under `#179`.

This internal tooling is not a new public electronic-structure method. It
consumes the converged native HF/KS endpoints rather than implementing SCF.

The [generated implicit-response adapter](implicit_response.md) reuses this
solver through an explicit callback. It generates transposed operators and source
weights from TensorIR rather than introducing a method-specific adjoint solver.

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

`NativeRKSResponse.from_native(batch, basis, grid=None, index=0)` connects the
actual successful native CPU LDA/PBE RKS state to this same operator and solver.
The optional explicit grid must exactly match the native points, weights and
owners. The optional `functional` must match the canonical SCF composition.
The adapter exports the native canonical orbitals, physical Fock, overlap,
occupations, energy and residual without rerunning SCF or recanonicalizing.
Its reference identity binds the native owner, model/provider, SCF domain and
solve/density/orbital generations. Direct unscreened CPU J/K uses the matching
native integral source. The batch and `NativeAO` must remain open; the adapter
owns its integral source and revocable snapshot lease.

```python
from tools.vibeqc_response import NativeRKSResponse, solve_many

# batch.execute(strict=True) has already converged; basis describes its exact AO source.
with NativeRKSResponse.from_native(batch, basis) as response:
    result = solve_many(response, rhs, strategy="recycled", raise_on_failure=True)
```

Native SCF quadrature includes low-density tails outside the legacy
`interior-v1` kernel domain. The native adapter differentiates the existing
scaled SCF point expression analytically in a restricted total-density
direction and feeds Cartesian potential coefficients into the same compact AO
assembly. It does not clip densities, omit tail points, finite-difference the
production potential, or substitute the interior-domain model. Exact vacuum is
accepted only with zero density/gradient direction. Undefined or unrepresentable
point derivatives fail the action. See [the SCF point domain](xc_scf_domain.md)
and [the binding decision](../.agents/notes/implemented/numerics/2026-09-20-native-rks-cpks.md).

Every action and solve validates the live lease, including zero RHS and blocked
zero-RHS solves that otherwise skip all actions. Replay (even of identical
geometry), failed replay, batch closure and response closure revoke old solves
and recycle spaces. Changed functional, grid, basis, provider or state are
rejected before publication.

These handoffs qualify all-electron CPU LDA/PBE RKS and UKS. CUDA CPKS, DF,
ECP, exact/range-separated exchange, and meta-GGA response remain unsupported.
AO/MO transforms, XC tiling and Krylov orchestration are host-side. Existing
solver workspace accounting is not a complete endpoint memory/performance
claim; the native kernel does not yet qualify implicit-response resource binding.
`tests/python/test_response_native_rks.py` checks independent libcint/Libxc
actions, finite orbital rotations, reconverged one-electron perturbations,
transpose/true residuals, multi-RHS/recycling, tails and lifecycle negatives.
The native `vibeqc_rks_response_tests` target also exercises the actual private
point-response ABI against 30 independent high-precision directions, its batch
layout and invalid-input boundaries, and energy snapshot leases from real
LDA/PBE H2 solves. See [point acceptance](xc_scf_domain.md#executable-evidence)
for the fixture generator and cancellation-aware numerical gate.

`NativeUKSResponse.from_native` uses the same arguments and lifetime contract
for the actual native CPU LDA/PBE UKS state. It preserves both canonical spin
frames and occupations. The existing spin reference/layout contract carries
an explicit `algorithm="UKS"` tag and functional/grid identities; the UHF
operator rejects this reference. `UKSResponseOperator` changes only the shared
spin operator's AO Fock-response seam: both outputs contain total Coulomb plus
their own XC response, including cross-spin correlation. The common XC tile
assembler and GMRES/multi-RHS/recycling code are reused without spin averaging.

An empty spin retains its zero-dimensional orbital-rotation block. Its point
density, gradient and response direction must be exactly zero; nonzero normal
directions are rejected because the exchange Hessian is singular there. The
other spin still has a nonzero response, including the correlation potential
in both output channels. This is a tangent-direction qualification, not a
finite full Hessian at the empty-spin boundary.

`tests/python/test_response_native_uks.py` uses real LiH+ and H2+ LDA/PBE states,
independent libcint/Libxc spin actions, finite orbital rotations, reconverged
spin densities, true residuals, transpose identities, multi-RHS/recycling and
lease/domain negatives. `vibeqc_uks_response_tests` checks 48 independent
high-precision point directions, spin permutations and the private batch ABI.
See [the spin binding decision](../.agents/notes/implemented/numerics/2026-09-20-native-uks-cpks.md).

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

`CudaDirectJKBackend` adapts the existing method-neutral `FockPlan` to the
shared RHF response operator. It requests exact full-Coulomb J/K, FP64 and
**zero screening**, preserving the `conventional-unscreened` Hamiltonian. It
copies the source's actual shell records rather than resolving a basis name.
The prepared CUDA provider is reused across signed, symmetric density actions;
raw J and K are returned without core-Hamiltonian contamination or additional
RHF factors. There is no CPU integral fallback or substitution of a DF operator.

The default `CudaDirectJKBackend` path remains host-orchestrated around CUDA
J/K contractions. B2 adds an opt-in `CudaResidentRHFResponse` owner for exact
conventional RHF. It borrows the same prepared direct-J/K source and CUDA
stream, retains C/orbital energies, Krylov vectors, AO response density/Fock
scratch and AO/MO transform scratch on device, and uses the existing #179 GMRES
control flow rather than a second solver. Vector copy/AXPY/dot/norm and
orthogonalization execute through cuBLAS; the direct J/K action is the existing
device-to-device provider seam.

The host still owns nuclear/metric RHS preparation, final occupied/density/
energy-weighted-density reconstruction, the small Hessenberg least-squares
problem, and convergence decisions. Thus diagnostics call this
`cuda-resident-host-controlled`, not an all-device CPHF. During a resident
operator action no density/J/K matrix crosses the PCIe boundary: only the
4-byte native numerical-status flag returns; dot/norm reductions return
scalars, and final solution publication is explicit. Directional Hessian
consumers suppress final Arnoldi-basis publication. Host preconditioners and
resident blocked-Arnoldi are not qualified and fail closed rather than falling
back to host execution.

The response device budget combines retained direct-J/K storage with the
resident owner allocation. It excludes provider preparation temporaries,
compiler/runtime metadata and CUDA-context/library-private memory. No speedup
is asserted from residency alone; complete solve/action/transfer timings remain
the relevant performance evidence.

The caller owns the `NativeSource` lifetime. Closed sources/backends, unrelated
geometry/basis/reference/Hamiltonian identities, nonsymmetric or nonfinite
inputs, unavailable CUDA and impossible device allocations fail explicitly.
Invalid results never increment successful action counts. The backend is
qualified for closed-shell RHF only; UHF/KS and molecular Hessian/HVP endpoints
are not enabled by its existence.

```python
import numpy as np
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_response import CudaDirectJKBackend, RHFResponseOperator

with NativeSource([(1, (0, 0, 0)), (1, (0, 0, 1.4))]) as source:
    reference, _ = export_rhf(source, backend="cpu", tolerance=1e-12)
    with CudaDirectJKBackend(source, device_budget_bytes=64 << 20) as backend:
        problem = RHFResponseOperator.build_problem(reference, backend)
        operator = RHFResponseOperator(problem, backend)
        action = operator.apply(np.ones(problem.dimension))
```

Run `tests/python/test_response_direct.py` for CPU-only ownership/identity and
failure contracts. In an explicitly allocated Slurm GPU job, set
`VIBEQC_RESPONSE_CUDA_TEST=1` and run
`tests/python/test_response_direct_cuda.py`. Device tests compare raw signed J/K
with committed independent AO-integral fixtures (including an f-shell case),
CPHF actions with explicit MO matrices and finite orbital rotations, and all
three shared multi-RHS strategies with independently evaluated true residuals.
One test binds an actual native RHF SCF snapshot. These checks are not complete
molecular-Hessian or all-device-solver acceptance.

See the [direct CUDA response decision](../.agents/notes/implemented/numerics/2026-09-19-direct-cuda-rhf-response.md).

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

The UHF operator remains HF-specific. Native UKS CPKS reuses its spin
layout and orbital-action implementation through the distinct UKS adapter
described above; the actual native KS state supplies its functional identity.

The direct CPU bridge can export a converged open-shell UHF solution through
`export_uhf`.  It canonicalizes the independently returned alpha and beta AO
densities, rechecks both physical commutators and density/Fock reconstruction,
and binds the result to the shared UHF response contract.  The bridge is
intentionally limited to the small direct CPU Hamiltonian: CUDA/DF UHF response
still fails closed until a spin-resolved device J/K response plan has separate
numerical and resource evidence. That gate is pinned by
`tests/python/test_response_uhf.py`, so neither the UHF CPU bridge nor the RHF
CUDA/DF backend is inferred as spin-resolved device support.


## Resident response failure and validation scope

Exceptional context exit destroys the resident native owner even when a solver
traceback still retains vector leases. The original solver error is preserved;
subsequent use of a retained vector rejects its closed owner. Ordinary explicit
`close()` still rejects live vector leases. Hardware-independent lifecycle
regressions cover memory, validation and runtime errors and idempotent teardown.

The resident CUDA numerical comparison in `test_cuda_runtime.py` requires an
explicit NVIDIA device allocation (`VIBEQC_RESOURCE_CUDA_TEST=1`) and skips under
`CUMETAL_ROOT`. The CuMetal workflow reports that skip; its green status is not
resident-response numerical qualification. NVIDIA compilation, host GMRES tests,
and ownership tests are distinct from executing the resident operator/solver
against the independent host-orchestrated CUDA reference.
