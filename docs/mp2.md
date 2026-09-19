# Canonical RHF-MP2 energy and conventional analytic forces

VibeQC exposes real FP64, closed-shell, all-electron conventional canonical RHF-MP2
on CPU and CUDA. Conventional four-centre execution provides total energies and
analytic nuclear forces; RI-MP2 provides energy only. This capability does not
include UMP2, ROHF-MP2, frozen core, ECP, complex orbitals, screened or
approximate correlation, Hessians, or mixed precision. No performance
replacement is implied by the numerical qualification.

```python
from vibeqc import Calculator

calc = Calculator(method="mp2", basis="sto-3g", device="cuda",
                  correlation_memory_budget_bytes=256 * 1024**2)
result = calc.singlepoint(
    [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))],
    properties=("energy", "forces"),
)
print(result.energy)       # total RHF + MP2 correlation, Hartree
print(result.forces)       # Hartree/Bohr; force = -gradient
print(result.correlation)  # OS/SS, reference, denominator, memory and transfers
```

RI-MP2 is selected explicitly. The auxiliary basis is part of the Hamiltonian;
when omitted, the orbital basis is used as the auxiliary basis.

```python
ri = Calculator(method="mp2", basis="sto-3g", device="cuda",
                density_fitting="auto",
                density_fitting_relative_threshold=1e-10)
result = ri.singlepoint(
    [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))],
    properties=("energy",),
)
```

RI-MP2 forces remain unsupported C2 work. An RI force request fails explicitly
without selecting conventional correlation or returning RHF-only forces. The C
ABI publishes the energy, complete force array, and correlation/response
diagnostic as one transaction; any reference, denominator, response,
derivative, CUDA, nonfinite, or output-buffer failure leaves caller storage
unchanged and invalidates prior diagnostics.

Homogeneous prepared MP2 batches support conventional energy and force
requests. Each item owns its reference, response, provider, diagnostics, and
candidate outputs; immutable method options alone are shared. A failed item
sets only its per-item status and does not overwrite its output storage or
poison successful neighbours or a later replay. MP2 batches explicitly reject
warm-start and HF profiling flags, RI force requests, invalid coordinates, and
unknown flags.

## Fixed mathematical contract

References are real FP64, canonical closed-shell RHF with every electron
correlated, occupied spatial columns first, 2/0 occupation, and a nonempty
virtual space. The existing all-electron Cartesian/real-spherical basis
support through f is used. Input geometries are Bohr; energies and orbital
energies are Hartree, gradients and forces are Hartree/Bohr, and
`force = -gradient`. Conventional reference and correlation use unscreened
exact Coulomb integrals. RI reference and correlation use the same auxiliary
basis, metric cutoff and fitted Coulomb Hamiltonian. MP2's default screening
is explicitly zero; a nonzero request fails rather than changing its
Hamiltonian.

For all **ordered** spatial indices i,j occupied and a,b virtual:

```text
g[i,j,a,b] = (ia|jb)                      chemists' ERIs
x[i,j,a,b] = (ib|ja)
D[i,j,a,b] = eps_i + eps_j - eps_a - eps_b < 0
E_OS       = sum g*g/D
E_SS       = sum g*(g-x)/D                alpha-alpha plus beta-beta
E_corr     = E_OS + E_SS = sum g*(2*g-x)/D
E_total    = E_RHF + E_corr
```

There is no further half/quarter factor in the restricted ordered-domain
formula. The independent spin-orbital oracle uses `1/4 |<IJ||AB>|²/D` and splits
occupied spin labels explicitly. Restricted amplitudes have simultaneous
`ijab↔jiba` symmetry, not independent occupied/virtual antisymmetry.

For RI-MP2, raw three-center integrals `A[mu,nu,P] = (mu nu|P)` and the
auxiliary Coulomb metric `M[P,Q] = (P|Q)` define
`B[mu,nu,Q] = sum_P A[mu,nu,P] M^(-1/2)[P,Q]`. Eigenvalues at or below
`density_fitting_relative_threshold * lambda_max(M)` are removed. The fitted
integral used in the same ordered MP2 equation is
`(ia|jb)_RI = sum_Q B[i,a,Q] B[j,b,Q]`.

Direct MO slots `(i,a,j,b)` and exchange slots `(i,b,j,a)` are separate provider
requests. Exchange is reordered into the same local ijab coordinates. Each
occupied axis has extent one; virtual extents use 1/2/4/8. Final virtual tiles
use zero coefficient columns and a valid virtual energy, so padding contributes
zero without introducing a zero denominator. The correlation consumer never
allocates complete molecular T2 or AO ERIs. CPU RHF preparation uses dense AO
ERIs with a checked capacity; CUDA RHF uses the bounded matrix-direct route.

## Native path and placement

```text
C / C++ / Python public prepare
  -> method registry -> Mp2Prepared
  -> existing RHF driver, requested owned physical reference
  -> conventional: CG10 RawSource + cyclic staged MO transforms
  -> RI: shared DF source + metric factor + occupied/virtual three-center transform
  -> CG08 energy equation -> generated native CPU or CG09 CUDA tile program
  -> native compensated scalar fold
  -> bounded MP2 adjoint + true-residual RHF Z-vector
  -> relaxed one-/overlap-/two-electron weights
  -> shell-local CPU or CUDA derivative contraction
  -> transactional energy + force + correlation/response diagnostics
```

Energy-only requests retain the existing correlation path and do not allocate
force state. Conventional force requests keep the physical RHF reference,
provider, adjoint, response workspace, relaxed weights, one shell-local AO
cotangent, derivative staging, and unpublished output candidates inside one
checked endpoint plan. They never materialize a molecular AO rank-four
cotangent, a coordinate derivative tensor, or a response-history tape.

CPU RHF uses the shared prepared Fock plan and iteration/DIIS/eigensolver. Its
values-only dense integral preparation counts both Cartesian representations
and the additional public tensor for spherical bases before allocation. This
limits the CPU reference size under the requested budget. It does not use the old 12-AO
Python exporter, reconstruct a second HF calculation or calculate HF forces.
GPU RHF explicitly uses the existing matrix-direct packed device evaluator,
never its small-system persistent-ERI mode. This generic exact device path is
usable without an sm_120 generated profile. The optimized quartet path retains
its strict generated/native shell-class coverage gate; that gate is not
weakened to make a portable build pass.

The final physical P,F(P),C,epsilon and S/h are exported as an owned reference.
CUDA column-major C is explicitly converted to CG10 row-major C[mu,p]. Native
host validation checks finite arrays, CᵀSC, FC=SCepsilon, canonical Fock,
commutator residual and canonical density drift. GPU computation of Fock,
orbitals and reference energy remains on device; its reference validation and
matrix snapshots are disclosed host staging.

Conventional correlation AO tiles come from the existing **CPU values-only
source**. CUDA mode uploads those tiles, performs all four cyclic transforms
with CG10 cuBLAS, downloads bounded MO tiles, reorders exchange on host, and
uploads them through CG09's host-input ABI.

RI mode builds the RHF reference and correlation integrals from the same
orbital/auxiliary systems and metric threshold. `density_fitting="cpu"` selects
CPU DF even when the calculator device is CUDA. `density_fitting="cuda"`
requires a CUDA context; `"auto"` follows the calculator device. CPU supports
auxiliary shells through g, while CUDA supports through f. The whitened
occupied-virtual three-center tensor replaces the four-center AO/MO transform
and is contracted into bounded energy tiles.

Each entire tile equation executes natively on the selected correlation
backend. The molecular loop and scalar fold are native C++, not Python
callbacks. CUDA execution includes disclosed host staging and is not an
entirely device-resident claim. There is no external quantum-chemistry
production backend or silent energy fallback. Generated plan symbols are
uniquely prefixed to coexist in one native library; all mathematical
coefficients come from the same TensorIR.

## Budgets, lifetimes and failures

`correlation_memory_budget_bytes` is an internal method numeric-capacity budget
composing sequential reference and correlation phases (zero means 256 MiB).
It is not a new global #203 planner, nor a process-RSS or free-VRAM guarantee.
The largest phase capacity is returned. Persistent reference/source buffers,
coefficient panels, two transform stages, detached/reordered MO feeds, tensor
arena, validation arithmetic, scalar outputs, library workspaces and retained
provider allowances are charged. Existing CG10 Python and native block plans
share `plan_spec.py` arithmetic; native code does not maintain a divergent
budget formula. Each CUDA transform is destroyed before the next is created.
RI admission also composes retained RHF/DIIS state with raw metric and
three-center owners, Cartesian-to-public transforms, metric factorization,
whitening and the occupied-virtual transformed tensor. Requests fail before a
phase whose declared numeric capacity exceeds the correlation budget.

CUDA HF uses the existing arena planner; provider handles receive explicit
retained allowances and actual queried solver host/device workspaces are
checked before allocation. Compact topology, reference validation/export and
temporary host matrices are counted conservatively. CUDA context/module/stack,
graph implementation metadata, allocator rounding, Python/C++ object headers
and BLAS host implementation overhead are outside the numeric accounting
scope. Device-wide observations are not treated as portable bounds.

Near-zero, nonnegative or nonfinite denominators fail explicitly, without
clamping/regularization. The default minimum magnitude threshold is `1e-10 Eh`.
Exactly degenerate occupied or virtual subspaces are accepted only when their
same-space MP2 Lagrangian derivative is zero at the same threshold; a
nonstationary degenerate subspace fails explicitly. SCF nonconvergence is
reported as reference failure; there is no fictitious MP2 convergence loop.
Bad reference, nonfinite integral/arithmetic, unsupported property/backend and
insufficient memory do not publish partial results.
Native context error details propagate to Python. Source/reference lifetime
is tied to the prepared system; changed geometry uses a newly prepared system
and cannot reuse an old reference or MO tile.

Force diagnostics append response iterations/restarts, true absolute and
relative residuals, response and derivative workspace bytes,
`planned_endpoint_peak_bytes`, `measured_endpoint_peak_bytes`, force provenance
flags, equation identity, and response-operator identity. `measured_response_workspace_peak_bytes` separately reports the actual
allocator high-water payload of GMRES-owned arrays, including its returned
solution; `response_workspace_allocation_count` counts successful allocations
in that domain. Both are zero for energy-only execution. The response-only
measurement excludes caller-owned inputs, operator-callback buffers, other MP2
stages, and allocator overhead. It must not populate the complete endpoint field.
A zero
`measured_endpoint_peak_bytes` means that endpoint allocation telemetry is
unavailable; it is not a zero-memory observation or a copy of the plan. This
slice currently reports that unavailable sentinel, so the measured-resource
qualification gate remains unsatisfied. Once measured, numeric ownership must
not exceed the simultaneous plan, which in turn must fit the configured numeric
capacity. CUDA transfer counters disclose staging; they are
not a claim that the full endpoint is device-resident. Older C callers receive
the supported struct prefix selected by `struct_size`.

## Qualification evidence and remaining issue scope

Separate tests cover fixed identical-C/ERI OS and SS, explicit spin sums,
permutations and rectangular tiles; native eight-loop AO→MO and independent RI
factor checks; VibeQC HF→public conventional/RI MP2 for H2/H2O/LiH/f-shell
fixtures; a 14-AO independent PySCF 2.14.0 conventional system exercising an
eight-plus-four virtual tail; bad states, nonconvergence, nonfinite arithmetic,
metric rank, mixed backend, denominator and budget boundaries; C/Python force
rejection and invalidation. Preserve the existing `1e-9 Eh` energy and
`atol=1e-11, rtol=1e-10` controlled component gates. PySCF is a test-only oracle.

The reproducible driver is `tools/validate_mp2_public_force.py`. It creates a
fresh directory, never overwrites prior results, and records exact source,
library, toolchain, hardware, model, response/resource diagnostics, independent
PySCF conventional MP2 analytic gradients, three fully re-solved Cartesian
finite-difference steps, translation/torque, rotational covariance, changed
geometry energy-force identity, batch replay, CPU/CUDA parity, and explicit
failure outcomes. The reviewed compact record is under
`benchmarks/results/issue193-conventional-force-b2/`; complete raw runs remain
in the persistent experiment location named by its manifest.
CPU runs may set `--fd-workers 4` to evaluate the same ordered plus/minus
displacements in isolated spawned processes; no displacement, step, SCF solve,
or acceptance gate is removed. CUDA qualification requires `--fd-workers 1`.

Compilation or a skipped CUDA test is not endpoint qualification. CUDA
evidence additionally includes real-device public single/batch execution and
compute-sanitizer results. Issue #193 B2 covers only conventional canonical
RHF-MP2 public forces. RI-MP2 forces remain unsupported C2 work and never
return HF or placeholder forces.
