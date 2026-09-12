# Fixed-density LDA/PBE XC integration (DFT03 first step)

DFT03 introduced internal CPU tooling that integrates an explicit fixed
density and assembles its AO potential. It reuses DFT01's native AO jets, grid
tiles and density features, and DFT02's audited expressions. The common
[contraction generator](xc_contractions.md) owns minimal point coefficients and
assembly; LDA skips gradient/tau reductions and GGA skips tau. Issue
[#162](https://github.com/jinzhezenggroup/vibeqc/issues/162) now also contains
separate native `LDA_RKS` and `PBE_RKS` vertical slices. The fixed-density
Python interface below remains the independent, more general LDA/PBE contract;
it does not itself run SCF, gradients or prepared GPU XC.

## Density, spin and energy conventions

Coordinates and AOs use the [DFT01 conventions](dft_grid.md), with Bohr lengths,
unit-normalized contracted Cartesian or real spherical AOs, ordinary spatial
jets `[value,x,y,z]`, and real symmetric matrices. The existing density validator
symmetrizes only accepted roundoff asymmetry (`atol=1e-12, rtol=1e-10`).
Input D need not be self-consistent or idempotent. No occupations, density,
electron count or weights are renormalized by the integrator.

| Input | Density interpretation | Returned potential |
| --- | --- | --- |
| `[AO,AO]` | Total D, Da=Db=D/2; RKS orbitals would give D=2 Cocc Cocc.T | `[AO,AO]`, derivative with respect to total D |
| `[2,AO,AO]` | Separate Da, Db, occupation factor one | `[2,AO,AO]`, derivatives with respect to each spin |

Both `polarized` and `unpolarized` functional layouts are supported. An
unpolarized expression requires exactly equal spin matrices after roundoff
symmetrization; unequal spins fail before integration. With separate equal
matrices it returns the same total-density derivative for each spin. With a
polarized expression and total D, the chain rule returns `(Va+Vb)/2`.

`LDA_XC_PW` means LDA exchange plus ordinary PW92 correlation; `PBE` means
`GGA_X_PBE + GGA_C_PBE`, whose correlation uses modified PW92 parameters.
These are the exact existing [functional definitions](xc_expressions.md), not
ambiguous aliases such as `LDA` or `SVWN`. Nonzero exact-exchange or range
metadata is rejected by this semilocal consumer.

The expression returns energy per volume `e_xc = (rho_a+rho_b)*epsilon_xc`,
in Hartree/Bohr^3. Libxc returns `epsilon_xc`; its independent PySCF consumer
multiplies by total rho. VibeQC must **not** multiply its `e_xc` by rho again.
Every grid weight already contains the full Bohr^3 measure, including radial,
angular and partition factors. Energy is `sum_p w_p e_xc(p)`.

For RKS integration, let `D=Da+Db` and `J[D]` denote the total-density
Coulomb matrix. The all-electron semilocal energy/Fock convention is

```text
E = E_nuc + Tr(D h) + 1/2 Tr(D J[D]) + E_xc
F_s = h + J[D] + V_xc,s
E_nuc = sum_{A<B} Z_A Z_B / |R_A-R_B|
```

The XC consumer returns only `E_xc` and `V_xc`; there is no Hartree half factor
in the XC term and `E_xc` is not `Tr(D V_xc)` or half that trace. A future global
hybrid with coefficient a adds `-a/2 sum_s Tr(D_s K[D_s])` to the energy and
`-a K[D_s]` to each Fock, using an explicitly matching semilocal composition.
No exact exchange is computed here.

## Matrix contraction and variational acceptance

For each spin, `rho_s = sum_mu,nu D_s,mu,nu phi_mu phi_nu` and
`sigma_ab = grad(rho_a) dot grad(rho_b)` with no cross-spin factor two.
The existing coefficient routine supplies

```text
G_a = 2 e_sigma_aa grad(rho_a) + e_sigma_ab grad(rho_b)
G_b = 2 e_sigma_bb grad(rho_b) + e_sigma_ab grad(rho_a)
V_s,mu,nu = sum_p w_p [e_rho_s phi_mu phi_nu
             + G_s dot (grad(phi_mu) phi_nu + phi_mu grad(phi_nu))]
```

For unpolarized XC, `G=2 e_sigma grad(rho)`. The generic assembly helper retains
DFT02's `(e_tau_s/2) grad(phi_mu) dot grad(phi_nu)` convention, tested using
synthetic nonzero tau derivatives; LDA/PBE tau derivatives are exactly zero.
This does not implement a meta-GGA.

Weights enter each bilinear once. The scalar term is not doubled when both
gradient legs are added. BLAS-sized panels avoid a point-by-AO-by-AO tensor;
final roundoff symmetrization is `(V+V.T)/2`. Tile contributions accumulate
without resetting E or V at a partial boundary.

For real symmetric perturbations, `delta E = sum_s Tr(V_s delta D_s)`.
Changing a diagonal entry has coefficient `V_ii`; changing both `D_ij` and
`D_ji` by h has coefficient `2 V_ij`. Tests exercise both, separate alpha/beta
and mixed-spin directions, and all three steps `1e-3, 3e-4, 1e-4`. Every step
must meet its explicit central-difference truncation bound; tests also require
contraction of the error curve. No favorable step is selected or hidden.

## Interface, identities and resource scope

```python
from vibeqc_compiler.dft import ExplicitGrid, NativeAO
from vibeqc_compiler.xc import FixedDensityXC, functional

xc = FixedDensityXC(functional("PBE", spin="unpolarized"))
grid = ExplicitGrid.read("my-explicit-grid.json")
with NativeAO(atoms, basis="sto-3g") as basis:
    result = xc.integrate(basis, grid, D, tile_points=251)
    E_xc, V_xc = result.energy, result.potential
```

The expression program alone is retained by `xc`. Every call reevaluates AO,
density and XC on the supplied grid; no numerical cache can hide a changed
grid, basis, density or functional. Results include immutable potential and
spin-electron-count arrays, complete input identities, point/tile counts and
the actual `cpu` backend. The density identity includes the total/separate
layout; the functional identity carries composition and expression provenance.

`MolecularGrid` streams fresh partitioned tiles and must match the basis atoms,
charge and spin policy. A moved basis with the old molecular grid fails. An
`ExplicitGrid` intentionally defines laboratory-frame points independent of
the atoms, allowing identical-grid tests and translations of AOs at fixed
points. Its content hash changes with points or weights. Rebuilding a molecular
grid is the caller's responsibility; do not label an old explicit laboratory
grid as a regenerated atom-centered grid.

Temporary storage scales with tile points B and AO count N: O(B*N) jets and
matrix panels, O(B*number_of_expression_nodes) interpreter intermediates, and
O(N^2) density/potential matrices. Molecular partition scratch additionally
scales as O(B*natom). Explicit grids are caller-owned O(npoint) input. This is
point tiling, **not a global memory-budget guarantee**. Native AO's existing
allocation guard remains in force; no second public memory-budget API or
J/K scheduler is introduced. #203 must account for XC graph intermediates,
matrix outputs, copies and allocator/BLAS overhead before a prepared method
claims a composed memory bound.

## Native LDA RKS vertical slice

The public method registry now advertises `LDA_RKS` with energy as its only
property. Its prepared CPU plan composes the existing conventional Coulomb
provider with a native materialized atom-centered grid and generated FP64
`LDA_XC_PW` evaluator. The default `GridSpec v1` is deterministic
`48 radial x 16 polar x 32 azimuth` per atom, using a rational Legendre radial
map, Legendre/trapezoid angular quadrature and equal-radius Becke partition.

Native SCF uses closed-shell occupations, `F=h+J+V_xc`, and the energy equation
above. Convergence requires energy change, density RMS and the physical
commutator residual to pass the requested thresholds; a depleted iteration
budget returns `VIBEQC_STATUS_NOT_CONVERGED` with diagnostics. The adapter
rejects odd-electron or non-singlet systems, forces, density fitting, auxiliary
bases, prepared batches and non-CPU contexts instead of falling back.

The native `lda-tail-v1` contract evaluates the exact positive-density formula,
uses the analytic zero-density energy/potential limit, and rejects negative or
nonfinite density. Its generated DAG uses the algebraically equivalent
`sixth-root-v1` parameterization so the smallest positive FP64 densities remain
finite without clipping or a density floor.

Current He/H2 endpoint numbers are also covered by an independent PySCF/Libxc
SCF consumer using the identical materialized `GridSpec v1` points and weights.
Both cases pass the initial absolute energy target of `1e-8 Eh`; this endpoint
acceptance is distinct from the already accepted fixed-density DFT03 fixtures.
The run is summarized in `build/issue-162-a/validation-summary.md` with remote
result SHA-256 `f4f85324ef505576e7231c4ead2056f775a4fefc464e7765a0dcc5d92cc93cf9`.

The native fixed-density layer now also contains an unpolarized PBE integrator
and AO potential. It uses the same spatial grid and conventional AO density,
including the GGA `sigma` contribution through first AO derivatives. Its
strict versioned `pbe-tail-v1` policy defines the exact vacuum zero and accepts
only the audited `interior-v1` density/reduced-gradient domain; it rejects
out-of-domain points rather than clipping them. The production
`pbe-tail-v2-lda-fallback` policy keeps exact PBE in that interior and uses the
stable positive-density `LDA_XC_PW` expression with zero sigma derivative in
the low-density/high-gradient tail. Native tests cover both contracts and the
tail-v2 finite-difference variational response.

The public `PBE_RKS` slice now reuses the existing CPU RKS SCF/J/DIIS path with
tail-v2, but remains provisional until an independent PySCF/Libxc matched-grid
endpoint is recorded. It is energy-only, closed-shell, conventional-J, CPU-only,
and has no prepared batch, forces, density fitting, UKS or CUDA support.

## Fixed-density domain and remaining dependencies

The original fixed-density Python consumer retains DFT02's `interior-v1`
first-derivative domain: vacuum, extremely small density, complete polarization
or other unsupported features raise `UnsupportedXC` with the failing tile
offset. This is intentionally distinct from native `lda-tail-v1`; neither path
clips density or skips zero-weight points. An empty explicit grid returns the
zero integral, not a converged quadrature. The small reference grids are fixed
controlled inputs, so agreement does not establish quadrature convergence.

Remaining #162 work includes LDA/PBE UKS, state invalidation and prepared CUDA.
The independent closed-shell PBE H2 endpoint is recorded in
`experiments/vibeqc/issue-162-a/pbe-rks-endpoint-20260912.md`; this does not
establish broader PBE coverage. #203 owns composed resource planning, and #163
owns nuclear gradients. Broader methods should
reuse the existing orthogonalization, provider and SCF infrastructure rather
than wrap full SCF in the Python fixed-density tooling loop.

## Independent reproduction

The exporter uses PySCF **2.14.0 / Libxc 7.0.0** `NumInt.nr_rks/nr_uks`, including
its own AO, density and matrix assembly. It receives identical explicit points,
complete weights, exact shell coefficients and supplied PSD matrices. No SCF
run or production XC helper is used by the reference. Cartesian AO scale S
from the independent overlap diagonal gives `D_pyscf=S D S` and `V=S V_pyscf S`.
Actual f shells in both representations exercise normalization and ordering.
Saved references include input/array/source hashes, library versions and the
small explicit AO cutoff (`1e-30`); ordinary tests load hash-checked data only.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VIBEQC_LIBRARY="$PWD/build/cpu/libvibeqc.so"
.venv/bin/python tools/generate_xc_integration_references.py /tmp/xc-reference-1
.venv/bin/python tools/generate_xc_integration_references.py /tmp/xc-reference-2
.venv/bin/python -m pytest tests/python/test_xc_integration.py -q
.venv/bin/python tools/validate_xc_integration.py --output /tmp/xc-integration.json
```

Compare every array hash from both generations before replacing fixtures.
Energy and every potential element use `atol=1e-11, rtol=1e-10`, retaining the
existing FP64 gate. Tests inject doubled density, doubled weights and doubled
potentials and require the independent gates to reject them. Weight scaling,
spin swaps, total/equal-spin equivalence, partial tiles, stale molecular grids,
changed basis/rules, invalid domains and symmetry are additional regressions.
The evidence runner records all matrix/energy errors and density finite
differences using #138's evidence envelope; it does not claim SCF acceptance.
