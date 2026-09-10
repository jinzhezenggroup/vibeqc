# Fixed-density LDA/PBE XC integration (DFT03 first step)

This internal CPU tooling slice integrates an explicit fixed density and
assembles its AO potential. It reuses DFT01's native AO jets, grid tiles and
density features, and DFT02's audited expressions. The common [contraction
generator](xc_contractions.md) now owns minimal point coefficients and assembly;
LDA skips gradient/tau reductions and GGA skips tau.
It does not register RKS/UKS, iterate SCF, implement nuclear gradients, or run
GPU prepared XC. It completes only this step of [#162](https://github.com/jinzhezenggroup/vibeqc/issues/162).

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

For future RKS integration, let `D=Da+Db` and `J[D]` denote the total-density
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

## Domain boundary and remaining RKS dependencies

All points, including zero-weight points, must satisfy DFT02's `interior-v1`
first-derivative domain. Vacuum, extremely small density, complete polarization
or other unsupported features raise `UnsupportedXC` with the failing tile
offset. No clipping, weight-based skipping or tail regularization is added.
An empty explicit grid returns the zero integral, not a converged quadrature.
The small reference grids are intentionally fixed controlled inputs; agreement
does not establish quadrature convergence or full molecular-grid tail support.

The next CPU RKS step can consume this tested density-to-E/V contract but still
requires a versioned, differentiated XC tail/limit policy, a native DFT method
adapter and native XC execution integration, common #202 Coulomb selection
(no exact exchange for semilocal XC), occupations/DIIS/residual evaluation,
convergence reporting, and method-level invalidation. Reuse existing HF
orthogonalization and iteration infrastructure; do not wrap full SCF in this
Python tooling loop. #203 owns composed resource planning. Public UKS, nuclear
gradients and GPU/prepared execution remain separate later work.

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
