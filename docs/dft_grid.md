# Atom-centered grids and spatial AO jets (DFT01)

`vibeqc_compiler.dft` is the compiler-side grid/AO/density interface installed beside the runtime.
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

## Production grid policy

`GridSpec(version=1)` remains the exact reference/fixture contract above. Production
KS defaults are resolved separately by the compiler-side `GridPolicy` into a
fully explicit `GridSpec(version=2)` before native execution. Modern LDA and
PBE/GGA use a 54×16×32 standard profile. `grid_accuracy="tight"`, and
the fixed-topology first-derivative profile, use 64×20×40 for LDA and 72×24×48
for GGA. Partition iterations remain three; pruning and screening remain
explicitly disabled so derivative topology does not change under response.

Production radii are the pinned `covalent_radius_bohr` values extracted from
`upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3/gfn1.json`. Their derived-table SHA-256 and upstream revision provenance
is attached only when the complete concrete v2 spec exactly matches a canonical
`GridPolicy` result. A user-constructed or deserialized v2 spec whose points,
radii or topology differ is identified as `explicit-grid-v2` and does not claim
xTBloom upstream provenance. The pinned table currently covers atomic numbers
1–86. A version-2 grid must carry a positive radius for every element it uses;
unsupported elements fail closed instead of silently receiving 1 Bohr.
LDA and PBE/GGA are the qualified version-2 policy families. Explicit
`GridPolicy.resolve` requests for meta-GGA/r2SCAN, VV10 or hybrids fail closed
until separately qualified. The KS options resolver preserves the already
qualified r2SCAN RKS/UKS default as `GridSpec(version=1)` when no explicit grid
is supplied and `grid_accuracy="standard"`. A nonstandard r2SCAN accuracy
profile requires an explicit `GridSpec`; this compatibility path does not
qualify r2SCAN for the version-2 production policy.

The resolved version, point counts, partition controls and complete radius table
enter the KS calculation payload and native snapshot identity. Serialization
therefore preserves the resolved contract rather than only an accuracy label.
Native descriptors that omit KS options retain the historical v1 behavior only
as an ABI/reference compatibility boundary; they are not a second production
policy. Production point counts are guarded by `benchmarks/grid_policy_convergence.py`:
independent PySCF SCF plus analytic grid-response gradients first verify that a
96×32×64 v2 reference is stable against 120×40×80, then bound standard/tight
energy and force error together with their deterministic point-count cost. The
PBE gate also retains the historical 48×16×32 candidate as a negative cost/accuracy
control; the promoted 54×16×32 profile adds radial resolution without the prior
18×36 angular-work expansion. See the [#596 grid-policy decision](../.agents/notes/implemented/architecture/2026-09-20-production-grid-policy.md).

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

## Generated moving-grid response (#163 B1)

`vibeqc_compiler.xc.grid_response.grid_response_tiles(grid, center_motion,
tile_points=...)` is a **CPU diagnostic/consumer building block**, not a complete
molecular gradient or a native CPU/CUDA force endpoint. It streams the same raw
atomic quadrature as `MolecularGrid.tiles`, then supplies separate point and
partition-weight directional derivatives. Element radial scales, atom ownership
and the unpruned topology are fixed; every partition center moves independently.

The common scalar `Graph` generates norm, Becke-switch, log-product and normalized
quotient JVPs. The CPU interpreter executes those generated roots. Pair reductions
retain log-domain stability and handle exact-zero product factors without
`0 * infinity`. Storage scales with `tile_points * atom_count`; no full
coordinate-by-grid Jacobian is constructed. Native resource-budget integration,
CUDA lowering and full KS assembly are not supplied by this interface.

Each response tile carries its source grid identity, displacement identity and
pair-branch identity, plus immutable `point_motion` and `weight_motion`. These
contract separately with the #163-A XC geometry partials. They do not prove that
an SCF state is current, and must not be relabelled as a complete stationary KS
force. The method consumer still owns the A-stage state gates and the remaining
one-electron/Hartree/Pulay/nuclear assembly.

For a multicenter grid, coincident centers at or below `coincident_tolerance`
and exact point/center collisions are rejected by this first derivative domain;
the existing value-only equal-split rule is unchanged. Pair clipping/zero-factor
branches are recorded. A branch identity is a diagnostic, not a proof that an
arbitrary finite displacement stays smooth: displaced validation must inspect
branch changes separately. There is no differentiation through pruning,
screening, radius changes, topology switches or SCF iteration history.

`tests/python/test_grid_response.py` checks point and center sources against
independent value-only finite differences and a 60-digit Decimal direct-product
oracle. It also checks rebuilt molecular weights, point/weight contractions,
translation, permutations, partial tiles, exact-zero factors, source identities
and fail-closed domains. These small fixtures do not assert broad molecule-scale
performance or authorize any public DFT force capability.

Rationale: [generated Becke response note](../.agents/notes/implemented/numerics/2026-09-19-generated-becke-grid-response.md).

### Mixed moving-grid response (#180)

The Hessian slice reuses the same scalar graph rather than differentiating the
first-response implementation numerically. `grid_mixed_response_program`
generates the primal, two independent JVPs and their mixed derivative for norm,
ratio, Becke switch and log primitives. `partition_mixed_response` composes
those roots through the normalized Becke product and keeps exact-zero factors
explicit: zero-, one- and two-zero product branches each use their correct mixed
limit. Left/right interchange is checked directly.

`grid_mixed_response_tiles` streams the corresponding first and mixed
partition-weight motions. Raw atom-centred grid points are affine in their owner
center, so point mixed motion is identically zero; no dense
coordinate-by-grid-by-coordinate tensor is formed. Independent tests compare
the mixed result with the finite difference of the existing analytic first
response on rebuilt molecular grids. This supplies the grid/partition geometric
primitive required by #180; it does not by itself publish a molecular DFT HVP.

### Semilocal XC Hessian bilinear (#180)

For LDA/GGA, `ContractionProgram(..., "geometry").mixed_geometry_directional`
combines the AO geometric JVPs with the existing generated XC feature gradient
and feature Hessian. The left direction is geometric; the right direction may
also carry the CPKS density response. For a discrete energy
`E = sum_g w_g e(z_g)`, it evaluates the mixed chain rule from first/mixed
measure motion, left/right/mixed feature motion and `d2e/dz2`. No third XC
feature derivative is required.

The routine returns the four separately auditable contributions from mixed
measure motion, the two measure-feature cross terms and the feature-mixed term.
It owns neither the CPKS solve nor molecular-grid motion policy: those remain
method-level #180 responsibilities. Meta-GGA is fail-closed here because its
density response has not been qualified.

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
points. Physical-atom motion can affect both and the partition.
`directional_ao_jets` applies the exact relative-coordinate chain rule to
existing jets without materializing an AO Jacobian. It consumes one additional
spatial derivative order, so LDA/GGA Hessian geometry can use the existing
through-order-3 native jet domain. The grid building blocks above supply
point/partition first and mixed motion; complete stationary HVP assembly remains
a method-level responsibility.

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
The [current-density source contract](density_sources.md) adds checked D/C
provenance, ragged/empty spin channels, local AO restrictions and output pruning
for CPU fixed-input consumers (#235 A).

## Prepared execution and budgets

```python
from vibeqc_compiler.dft import GridSpec, PreparedGrid

with PreparedGrid(
    atoms, basis="sto-3g", spec=GridSpec(), tile_points=251, budget_bytes=256 << 20
) as grid:
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

The [current-density source contract](density_sources.md) adds bounded GPU
D/C routes and an explicit GPU-density/CPU-XC E/V adapter (#235 A/B), including
source stamps, availability fallback and composed numeric resource budgets.

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


## Complete stationary CPU consumer

The generated grid-response building block is consumed by the internal
[complete native-state CPU RKS gradient diagnostic](ks_diagnostics.md#native-cpu-stationary-gradient-diagnostic).
That consumer binds the physical native atomic measures, point coordinates,
D/F/C/W and grid identity, and adds all remaining molecular sources. The grid
module alone is still not a complete gradient, and neither interface enables
public DFT forces or qualifies native CUDA grid-response execution.


An explicit [native CPU grid adjoint](stationary_native_consumers.md) now
consumes the same scalar norm/ratio/log/Becke graphs and raw atomic measures.
It contracts all nuclear coordinates with two pair passes per point instead of
repeating the directional interpreter for each coordinate. Native execution,
component budgets and output failure gates are distinct from full public
DFT-force or CUDA qualification.
