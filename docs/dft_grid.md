# Atom-centered grids and spatial AO jets (DFT01)

`tools.vibeqc_dft` is the internal source-checkout grid/AO/density interface.
It provides CPU quadrature, native CPU/CUDA AO derivatives and spin density
features for later XC/SCF consumers. It registers no DFT method, XC functional,
nuclear gradient or Hessian. Use `PYTHONPATH=.:python` and a built native
library. PySCF is needed only to regenerate independent fixtures.

## Exact grid prescription

`GridSpec(version=1)` records every numerical rule:

| Field | Default meaning |
| --- | --- |
| Radial rule | 48-point Gauss–Legendre in `t∈(0,1)`, `r=R*t/(1-t)` |
| Radial weights | `r² dr = R³*t²/(1-t)^4 dt` |
| Angular rule | 16-point Gauss–Legendre in `cos(theta)` × 32 equally spaced azimuths |
| Angular weights | `w_polar*2π/n_azimuth`, summing to `4π` |
| Element radii | Exactly 1 Bohr; explicit `(atomic_number,R)` overrides |
| Partition | Equal-radius Becke, three compositions of `(3μ−μ³)/2` |
| Coincidence | Pair distances ≤`1e-12` Bohr get equal pair ownership |
| Pruning and screening | Disabled |
| Units and ordering | Bohr; atom, radial, polar, azimuth |

This tensor-product angular rule derives directly from polynomial quadrature;
it is not a Lebedev table. There are no imported angular-table or empirical
radius licenses and no runtime downloads. The partition follows A. D. Becke,
J. Chem. Phys. 88, 2547 (1988), with no heteronuclear radius correction.
Radius overrides change the radial grid, not the partition. All these choices
participate in the identity; an opaque accuracy level is insufficient.

`MolecularGrid` retains radial/angular topology and streams bounded tiles. It
computes normalized ownership using log products. Coincident atoms share
ownership, avoiding duplicate molecular measure. Points move with their owning
centers, and partition weights are recomputed for every new geometry. Tests
cover far-separated, near-coincident and reordered atoms. Partition of unity
alone is not a quadrature accuracy guarantee.

`grid.explicit(max_points=...)` is a guarded small-grid exporter. `ExplicitGrid`
stores exact points, weights, owners, provenance and a verified content hash.
Independent tests pass identical unpartitioned atomic data to PySCF, compare
its native partition weights, then compare every AO/feature on identical
points. Separate refinement tests integrate known Gaussian/Slater functions
and molecular densities approaching `Tr(DS)`. No weight or density
renormalization conceals quadrature error.

## AO derivative conventions

`NativeAO` owns normalized shell state after the original system handle is
released. Primitive normalization and real-spherical transformations come
from `src/molecule/basis.cpp`. Cartesian AOs use CCA/libcint order; spherical
d/f AOs use the existing PySCF real-harmonic order. Public through-f bases are
supported; higher angular momenta fail explicitly.

Output is `[jet,point,AO]`. `jet_indices(order)` enumerates ordinary spatial
derivatives through order zero, one, two or three:

```text
0: value
1: x, y, z
2: xx, xy, xz, yy, yz, zz
3: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
```

There is no factorial scaling. Mixed derivatives occur once; contractions
with a full symmetric Hessian must supply off-diagonal multiplicities.
Spatial differentiation holds centers fixed. Moving a basis center at a fixed
point gives minus its spatial derivative. Moving a grid owner translates its
points. Physical-atom motion can affect both and the partition; complete
nuclear derivatives need that separate chain rule, which is not provided here.

The CPU differentiates polynomial coefficients recursively. CUDA independently
uses the Leibniz rule with closed Gaussian derivatives. Temporary powers can
reach six for differentiated f functions without exposing a new public basis
angular momentum. Exact exponential underflow contributes zero. No AO
magnitude cutoff is applied, and nonfinite outputs fail explicitly.

## Density features

An RHF matrix is the **total** density and splits equally into alpha and beta.
Separate spin input has shape `[2,AO,AO]`. Matrices may be arbitrary non-SCF
states; they must be finite, real and symmetric within `atol=1e-12, rtol=1e-10`.
Accepted roundoff asymmetry is symmetrized. Negative diagnostic densities are
preserved; complex data are unsupported.

| Feature | Definition and output layout |
| --- | --- |
| rho | `Σμν Dσμν χμ χν`, `[spin,point]` |
| grad rho | `2 Σμν Dσμν χμ grad χν`, `[spin,point,xyz]` |
| sigma | `(grad rho_a², grad rho_a·grad rho_b, grad rho_b²)`, `[aa/ab/bb,point]` |
| tau | `1/2 Σμν Dσμν grad χμ·grad χν`, `[spin,point]` |

Cross-spin sigma has no extra factor two. `orbital_features` independently sums
supplied occupied/fractionally occupied orbitals without constructing D.
Fixtures also check the Laplacian contraction using second AO derivatives.

## Prepared execution and budgets

```python
from tools.vibeqc_dft import GridSpec, PreparedGrid

with PreparedGrid(atoms, basis="sto-3g", spec=GridSpec(),
                  tile_points=251, budget_bytes=256 << 20) as grid:
    first = grid.integrate(density)  # spin electron counts and integrated tau
    grid.reconfigure(coordinates=new_coordinates)
    moved = grid.integrate(new_density)
```

`iter_features` exposes bounded tiles to later consumers. All AOs remain active
in the unscreened route; points are tiled. Resident D scales as O(NAO²), tile
jets as O(njet*npoint_tile*NAO), and partition scratch as O(npoint_tile*natom).
A complete molecular grid-by-AO-by-jet array is never required.
`NativeAO.evaluate` additionally supports partial AO slices for validation.

CUDA selection is explicit and requires an artifact from
`compile_cuda(CudaCompilerAdapter(...), cache)`, using the shared finite NVCC
adapter and verified native-runtime cache. The source also compiles in the
normal CUDA library/CI. A private stream, cuBLAS handle, normalized basis, two
spin matrices, AO tile, matrix-product panels and outputs remain resident.
AO evaluation, matrix contractions and sigma arithmetic run on device. Grid
construction/partition remains on the host; points upload and features download
explicitly. Validation can request detached AO jets. Ordinary streams are used.

`TilePlan` checks capacities before native GPU allocation/evaluation, charging
basis copies, D validation, points/owners, jets, matrix work, output copies,
partition buffers and radial/angular setup scratch. CUDA adds a 4 MiB explicit
workspace and the established, separately checked 96 MiB cuBLAS allowance.
Actual arena bytes must equal the plan. Host basis/grid setup precedes the
final capacity check; this is not a preflight allocation guard on arbitrary
host topology, although setup scratch is charged. GPU allocation follows the
check.

These are numeric-buffer bounds. Object headers, allocator rounding, internal
BLAS host workspace and CUDA context/modules/stacks are outside scope. Caller
retention of multiple detached tiles or suspended invalidated iterators adds
host memory beyond the single-output publication allowance. Transactional
updates check **old plus replacement**
capacity before allocating replacement GPU state. A tight budget may require
an explicitly larger update budget or closing/recreating the plan. Diagnostics
report the transient overlap separately. This is not a total-VRAM bound.

Geometry, grid rules, basis coefficients, representation, atom order and
explicit charge/spin policy enter identities. Every reconfiguration advances
the generation. Failed updates preserve usable old state. Starting another
iterator invalidates earlier iterators, preventing interleaved GPU density
uploads from contaminating results. Complete integrations serialize per plan.

`PreparedGridBatch` charges summed capacities and preserves explicit ragged
point/AO offsets. Invalid density items fail independently. Use
`batch.reconfigure(index, ...)` to update an item under the fleet budget and
recompute offsets. Plans execute sequentially on private streams; this initial
wrapper does not claim a fused GPU batch launch.

## Validation and reproduction

`tools/generate_grid_references.py` pins PySCF 2.14.0, with exact geometries,
original coefficients, densities, orbital factors, units and array hashes.
Six fixtures cover H2, asymmetric water, actual Cartesian/spherical f shells,
diffuse and tight exponents. CI loads fixtures without importing PySCF.
PySCF (Apache-2.0) and libcint (BSD-2-Clause) are test references; no external
solver source is copied.

```bash
PYTHONPATH=.:python VIBEQC_LIBRARY=$PWD/build/cpu/libvibeqc.so \
  python -m pytest tests/python/test_grid_cpu.py tests/python/test_grid_prepared.py -q

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env PYTHONPATH=.:python VIBEQC_LIBRARY=$PWD/build/cpu/libvibeqc.so \
  VIBEQC_GRID_CUDA_TEST=1 OMP_NUM_THREADS=1 \
  python -m pytest tests/python/test_grid_cuda.py -q

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:15:00 \
  env PYTHONPATH=.:python VIBEQC_LIBRARY=$PWD/build/cpu/libvibeqc.so \
  OMP_NUM_THREADS=1 python tools/validate_grid.py --cuda --output /tmp/grid-evidence
```

Here the CPU library supplies normalization; the explicitly compiled artifact
executes CUDA AO kernels and cuBLAS. The evidence CLI separates identical-grid
element gates from quadrature convergence. It records cold setup, fixed replay,
geometry rebuild, ragged throughput, transfers, compilation/resources and
numeric capacities, with repeated interleaved CPU/CUDA samples. It does not
automatically promote a schedule or assert DFT energy/force performance.

The [DFT01 evidence archive](../benchmarks/results/dft-grid-160/README.md)
preserves the clean scientific revision, raw numerical/timing records,
sanitizer logs, actual loaded-library versions and capacity scope.
