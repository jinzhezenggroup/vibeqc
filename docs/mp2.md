# Canonical RHF-MP2 energy

The #193 A1 candidate adds public conventional MP2 energy-only preparation.
Its current validation status and frozen source identity are recorded in the
handoff; the whole #193 issue also requires RI energies and complete gradients.
No gradient, RI, frozen-core, open-shell or ECP capability follows from this
conventional energy implementation. No performance replacement is promoted.

```python
from vibeqc import Calculator

calc = Calculator(method="mp2", basis="sto-3g", device="cuda",
                  correlation_memory_budget_bytes=256 * 1024**2)
result = calc.singlepoint([("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))])
print(result.energy)       # total RHF + MP2 correlation, Hartree
print(result.correlation)  # OS/SS, reference, denominator, memory and transfers
assert result.forces is None
```

`singlepoint(..., properties=("energy", "forces"))` rejects MP2 before
executing. The C ABI rejects a non-null force request without writing an
output. The C++ `Calculation` wrapper defaults to energy-only and represents
absent forces with `std::optional`; the PyTorch analytic-backward wrapper
explicitly requests forces and therefore rejects MP2. Prepared batches are
not enabled by this slice. Independent single-system plans own independent
state; calling the same plan recomputes its physical reference. A failed
execution invalidates its previous correlation diagnostics.

## Fixed mathematical contract

References are real FP64, canonical closed-shell RHF with every electron
correlated, occupied spatial columns first, 2/0 occupation, and a nonempty
virtual space. The existing all-electron Cartesian/real-spherical basis
support through f is used. Input geometries are Bohr; energies and orbital
energies are Hartree. Conventional reference and correlation use unscreened
exact Coulomb integrals. MP2's default screening is explicitly zero; a
nonzero request fails rather than changing its Hamiltonian.

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
  -> CG10 RawSource + native adapter of cyclic staged MO transforms
  -> CG08 energy equation -> generated native CPU or CG09 CUDA tile program
  -> native compensated scalar fold -> energy + correlation diagnostics
```

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

Correlation AO tiles come from the existing **CPU values-only source**.
CUDA mode uploads those tiles, performs all four cyclic transforms with CG10
cuBLAS, downloads bounded MO tiles, reorders exchange on host, and uploads them
through CG09's host-input ABI. Each entire tile equation executes natively on
GPU. The molecular loop and scalar fold are native C++, not Python callbacks.
This is a bounded mixed-placement path, not an entirely device-resident claim.
There is no external quantum-chemistry production backend or silent CPU energy
fallback. Generated plan symbols are uniquely prefixed to coexist in one
native library; all mathematical coefficients come from the same TensorIR.

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

CUDA HF uses the existing arena planner; provider handles receive explicit
retained allowances and actual queried solver host/device workspaces are
checked before allocation. Compact topology, reference validation/export and
temporary host matrices are counted conservatively. CUDA context/module/stack,
graph implementation metadata, allocator rounding, Python/C++ object headers
and BLAS host implementation overhead are outside the numeric accounting
scope. Device-wide observations are not treated as portable bounds.

Near-zero, nonnegative or nonfinite denominators fail explicitly, without
clamping/regularization. The default minimum magnitude threshold is `1e-10 Eh`.
SCF nonconvergence is reported as reference failure; there is no fictitious
MP2 convergence loop. Bad reference, nonfinite integral/arithmetic, unsupported
property/backend and insufficient memory do not publish partial results.
Native context error details propagate to Python. Source/reference lifetime
is tied to the prepared system; changed geometry uses a newly prepared system
and cannot reuse an old reference or MO tile.

## Validation and remaining issue scope

Separate tests cover fixed identical-C/ERI OS and SS, explicit spin sums,
permutations and rectangular tiles; native eight-loop AO→MO checks; new
VibeQC HF→public MP2 for H2/H2O/LiH/f-shell fixtures; a 14-AO independent PySCF
2.14.0 system exercising an eight-plus-four virtual tail; bad states,
nonconvergence, nonfinite arithmetic, denominator and budget boundaries; C/Python
force rejection and invalidation. Preserve the existing `1e-9 Eh` energy and
`atol=1e-11, rtol=1e-10` controlled component gates. PySCF is a test-only oracle.

Full CPU/real-GPU, sanitizer, source/library identity and device records must
be read from the candidate's evidence, not inferred from test definitions.
Compile success or an optional skip is not GPU acceptance. Fast-compile builds
are explicitly marked and have no performance-promotion claim.

RI energy requires a separately declared Hamiltonian/auxiliary basis/metric
contract. Complete conventional gradients depend on the actual #151/#179/
#141/#144 interfaces; RI gradients additionally need #143. Those missing
derivatives remain unsupported and never return HF or placeholder forces.
