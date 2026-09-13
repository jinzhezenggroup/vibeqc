# Current-density D/C feature sources (#235)

`vibeqc_compiler.dft.DensitySource` provides a fixed-input CPU contract for
choosing between the original density matrix D and compatible occupied or
fractionally occupied orbitals C/f. Slice A (#295) defines source validity;
slice B (#297) adds bounded native CUDA execution below.
It builds on the [grid feature conventions](dft_grid.md) and supplies
the same feature dictionary consumed by XC contractions.

## Mathematics and spin conventions

For each real spin block, define `B = C sqrt(f)` with nonnegative occupations:

```text
D = B B.T
Y = Phi D                  Psi = Phi B
rho = sum(Phi * Y)          rho = sum(Psi**2)
grad_k rho = 2 sum(Phi_k * Y)
                           grad_k rho = 2 sum(Psi_k * Psi)
tau = 1/2 sum_k sum((Phi_k D) * Phi_k)
                           tau = 1/2 sum_k sum(Psi_k**2)
```

Summations reduce the complete AO or orbital dimension before forming
`sigma = (grad_a², grad_a dot grad_b, grad_b²)`. Cross-spin sigma has no
additional factor two. A total matrix `[AO,AO]` splits equally between alpha
and beta. Orbital inputs always carry **per-spin** occupations; an RHF total
occupation of two becomes one in each spin channel. Occupations are never
renormalized or inferred from electron count. B is formed before collocation
so a zero occupation does not multiply an overflowing unweighted square.

Both rectangular `[2,AO,norb]` inputs and pairs of `[AO,norb_spin]` arrays work.
An empty spin channel has shape `[AO,0]` and occupation shape `[0]`. Orbital
columns are not truncated. An explicit `ao_ids` map restricts D to `D[I,I]`
and C to `C[I,:]` in the same order, retaining all cross terms and occupied
columns. Unsorted unique maps and empty local supports are supported.

## Identity, validation and replay

```python
from vibeqc_compiler.dft import DensitySource

# basis.identity includes coordinates, normalized basis, AO representation,
# atom order, charge and spin policy. D is this producer state's density.
current = DensitySource(
    D, basis_identity=basis.identity,
    basis_generation=3, density_generation=17,
)
candidate_stamp = current.stamp  # capture with C/f at their production time
checked = current.with_orbitals(C_spin, f_spin, stamp=candidate_stamp)
features = checked.features(
    jets, stamp=current.stamp, route="orbitals",
    ingredients=("rho", "gradient", "sigma"),
)
```

`DensityStamp` identifies the normalized D content and total/separate layout,
basis content, basis generation, density generation, and state/response role.
The existing `spin_densities` validator preserves signed D and symmetrizes only
accepted roundoff asymmetry. Every array has its own immutable backing bytes;
new sources with attached factors share the immutable original D. An accepted
`factor_identity` hashes both C/f spin blocks, their shapes and their stamp.

The caller must carry the stamp with the orbital snapshot and supply the
**current consumer's** stamp on replay. An old stamp cannot be relabeled merely
because dimensions match. Same-density producer updates still advance the
density generation; changed geometry advances the basis generation. Creating
a new source computes a content hash as a further check on changed matrices.
There is no trusted-producer bypass in this external-input CPU contract.

Attachment compares every entry of `B B.T` to the actual D, using
`atol=1e-12, rtol=1e-10`. This is numerical compatibility at the existing
density tolerance, not exact-arithmetic equality or a universal bound on
subsequent XC errors. Compatibility uses row panels of at most
`validation_rows` (default 64); it costs O(NAO²*norb) once per attachment.
`validation_max_abs_error` reports the largest accepted entry error. Replay
checks the stamp without reconstructing D. This CPU reference deliberately
does not infer an eigendecomposition or repair eigenvalues to manufacture C.

`with_orbitals` returns a new source. Rejected candidates clear any previously
accepted factors; the original source remains usable. Diagnostics distinguish
`missing_orbitals`, `stale_orbitals`, `invalid_orbitals: ...`,
`incompatible_orbitals` and `response_density`. Invalid coefficients include
complex/nonfinite data, shape mismatches and negative occupations. A source
declared `role="response"` always keeps D, including when a particular response
happens to be positive. Missing or rejected C never changes the original D.

`route="auto"` selects by availability in this CPU reference only: validated
C, otherwise D. It is not a cost model or production performance promotion.
Explicit `density_matrix` and `orbitals` routes support parity tests; forcing
an unavailable orbital route raises with its fallback reason. Replaying a
stale **source** raises instead of falling back to its equally stale D.

## Requested outputs, storage and integration boundary

Both `density_features` and `orbital_features` accept `ingredients` drawn from
rho, gradient, sigma and tau. Rho alone accepts value-only AO jets; requesting
sigma computes the needed gradient internally. Tau requires first derivatives
and is omitted entirely when unused. Ordinary derivative orders 0–3 remain
supported within the requested domain. Higher supplied jets are not contracted.

Retained storage includes both spin D matrices, C and occupations. Attachment
additionally holds one weighted factor O(NAO*norb) and validation temporaries
O(validation_rows*NAO), including matrix/error/comparison panels. Tile execution
uses local D or C gathers, input-validation copies, weighted factors, and up to
four point-by-orbital panels per spin. The caller supplies a bounded point tile;
this helper never assembles a complete molecular grid or stores tile history.
It is not a composed memory-budget guard, and does not remove D storage.

Native `src/scf/density_factor.hpp` already owns the SCF/RI-K factor contract:
its integer occupations, exact density witness, native reference/orbital IDs
and restricted-spin convention remain authoritative there. This CPU external
reference adds fractional-spin mathematical acceptance without changing native
SCF semantics. A future #235 C adapter must reuse that producer provenance
and map restricted occupations explicitly, then compose resources under #203.
Native SCF integration, geometric derivatives and candidate
selection under #168 remain slice C. Spatial/prepared integration is described
below. Neither A nor B supplies a new molecular
method, solver, force capability or speedup claim.

## Bounded native CUDA execution

`CudaGrid` uses the same AO owner, cuBLAS adapter, stream and generated
bilinear density formulas for D and C. Prepare `orbital_capacity=(na, nb)`
and `orbital_tile`, then call `set_source(source, stamp=current, route=...)`.
Every subsequent feature `evaluate` or `task` requires the current stamp.
Basis content and basis generation must match the immutable prepared owner.
Uploads bind D and the validated weighted factors together; a failed transport
leaves features unavailable until another successful upload. Ordinary
`set_density` explicitly clears source/orbital state and supports signed D.

The C route packs each selected AO row and each occupied-column tile, computes
bounded Psi/derivative panels with cuBLAS, and reduces features on the GPU.
It visits every supplied column, including zero-occupation columns; it does
not assume local AOs make delocalized orbitals sparse. Sigma is formed after
all tiles in both spins, retaining cross terms. Empty spins/maps/point tails
are valid. Local CUDA maps follow the existing sorted-unique device ABI.

`route="auto"` is availability-based: missing/rejected factors or insufficient
prepared orbital capacity execute the original D. `source_statistics` reports
the route, factor/source identity, fallback reason, occupied counts, weighted
factor packing time and upload bytes/time. Explicit `orbitals` rejects an
unavailable route. This policy is not the #168 cost selector. Response sources
always retain D; C geometric derivatives are not exposed.

Requested ingredients prune arithmetic and unnecessary GEMMs. Rho-only accepts
order-zero AO jets; sigma requires gradients internally; tau is optional. The
transfer ABI retains thirteen slots, while Python publishes only requested
arrays. Borrowed device task ABI v1 describes the full feature layout and
rejects pruned requests. Calls sharing an owner serialize; independent owners
have separate streams and memory.

Pass `resource_budget=ResourceBudget(...)` for composed #203 preflight, or
the legacy `budget_bytes`, but not both. The planner charges global D, bounded
AO/work/output tiles, global B, active-AO-by-orbital packing, a Psi capacity
of `8 * 4 * tile_points * orbital_tile` bytes, host copies and a cuBLAS allowance.
Psi capacity is independent of the total occupied count. Native arena bytes
are checked against the plan and provider observations against the allowance.
This is a conservative numeric capacity bound, not measured process peak RSS
or total GPU-context residency. Python objects, allocator rounding, CPU BLAS
internals, CUDA context/module/stack costs beyond the explicit allowance,
caller-owned source construction/validation and retained output history are
excluded and require separate accounting. D storage remains necessary.

## Fixed-density XC adapter and evidence

`PreparedXCContractions(..., density_grid=cuda, resource_budget=budget)` borrows
a dense `CudaGrid` with the matching basis, AO order and required ingredients.
Call `execute(source, stamp=current, route=...)`. The adapter holds the CUDA
owner's lock through the entire execution and composes its resource request
with existing XC workspace and global potential output capacity.

AO and density features execute on GPU. Each tile explicitly downloads AO jets
and features for the existing generated native **CPU** XC point code and AO
potential assembly. `collocation_backend="cuda"`, `xc_backend="native_cpu"`,
device phase metrics, CPU contraction time, source packing/upload costs and
whole-call timings expose this boundary. Native CPU `energy`, response and
geometry APIs remain available through their existing adapters; this CUDA
adapter supports fixed-density energy **plus potential** only. It does not
claim device-resident XC, native SCF integration or complete forces.

`tests/python/test_density_cuda.py` covers six independent feature fixtures,
all ingredient subsets, fractional/empty spins, point/orbital tails, local
cross terms, source invalidation/fallback, owner isolation, and 48 same-grid
LDA/PBE E/V combinations at 128/256 MiB. Every entry uses the FP64 gates above.
The positive-definite integration fixtures use test-only Cholesky factors,
without clipping; factorization and external validation are outside replay.

Reproduce tests and diagnostic endpoint/resource evidence from a clean source
checkout with a matching Release CPU normalization library. Use fresh cache
and output paths for the benchmark. PySCF need not be installed.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VIBEQC_LIBRARY=$PWD/build/cpu/libvibeqc.so VIBEQC_GRID_CUDA_TEST=1 \
  VIBEQC_NVCC=/group/software/cuda-12.9.1/bin/nvcc \
  .venv/bin/python -m pytest tests/python/test_density_cuda.py tests/python/test_grid_cuda.py -q

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VIBEQC_NVCC=/group/software/cuda-12.9.1/bin/nvcc \
  .venv/bin/python tools/benchmark_density_sources.py \
  --library build/cpu/libvibeqc.so --cache .artifacts/density-bench-cache \
  --output .artifacts/benchmarks/density-sources --samples 5
```

The benchmark records five interleaved D/C pairs per endpoint case, cold
compilation, owner construction, source/fixture-factor/validation costs and
full available E/V wall time. Small fixtures and CPU staging do not establish
a performance winner; larger molecular/throughput endpoints and promotion
remain with #235 C/#168. Publish reviewed numerical records through the
[existing evidence envelope](evidence_retention.md).

## Validation

`tests/python/test_density_source.py` consumes the existing hash-checked PySCF
fixtures for H2, water, Cartesian/spherical f shells, diffuse and tight bases.
It checks every rho/gradient/sigma/tau element on identical saved AO jets at
`atol=1e-11, rtol=1e-10`; no fixture regeneration or PySCF import is required.
Further tests cover fractional/empty spins, signs and equal-occupation rotations,
three-step density directional differences, local cross terms, AO nodes,
zero/tiny occupations, invalid and stale factors, immutable state and replay.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/python/test_density_source.py -q
```

## Prepared spatial integration (#235 C1 / #299)

`PreparedSpatialGrid(..., backend="cuda", orbital_capacity=(na, nb),
orbital_tile=..., ingredients=..., basis_generation=...)` prepares the same
bounded D/C CUDA owner behind the existing spatial task API. Pass the current
`DensitySource`, `stamp` and optional `route` to `iter_features` or
`device_tasks`. Plain density arrays retain their original D-only behavior;
the CPU spatial adapter continues to accept plain D. Source construction and
external factor validation remain caller-owned setup costs.

The spatial certificate still covers complete through-order AO jets with at
least first derivatives. Prepared `ingredients` prune CUDA feature arithmetic;
a diagnostic consumer may publish a subset of those ingredients or jets.
That publication subset does not change the prepared arithmetic or AO mask.
Device task ABI v1 still requires all four feature outputs. Every new execution
uploads current state once, invalidates old diagnostic iterators, and starts
a fresh global device potential for subsequent lease scatters. Transport
failures invalidate old executions and clear source diagnostics.

```python
with PreparedSpatialGrid(
    basis, grid, backend="cuda", artifact=artifact,
    policy=policy, orbital_capacity=(na, nb), orbital_tile=3,
    ingredients=("rho", "gradient", "sigma"), resource_budget=budget,
) as spatial:
    with PreparedXCContractions(
        native_pbe, basis, grid, spatial=spatial, resource_budget=budget,
    ) as endpoint:
        result = endpoint.execute(source, stamp=current_stamp, route="orbitals")
```

This XC adapter borrows the spatial owner's CUDA buffers, recomputes each
selected AO tile, downloads its jets/features, and assembles/scatters local
potential blocks on the CPU using the same fixed mask. The observable remains
fixed-density energy plus potential. CUDA response and geometry requests are
rejected. Full native SCF, complete forces and automatic candidate selection
remain #162/#163/#168 work.

The composed budget charges spatial metadata, D/B, bounded AO/orbital panels,
the cuBLAS allowance, CPU XC workspace and global potential output. The borrowed
CUDA arena is charged once. Transactional reconfiguration must fit old and new
owners simultaneously; failure preserves usable old state. A replacement
invalidates existing CUDA XC consumers even when the mask is identical.
Statistics retain the source decision, upload/packing and device phase times,
whole-call/CPU cost, mask identity, per-task active AO counts and matrix products.
They distinguish numeric capacity from measured whole-process peak memory.

`tests/python/test_spatial_density_cuda.py` checks global-D zero-masked oracles,
fractional/empty spins, orbital counts larger than local AO supports, both
AO representations through f, lease/reset/replacement lifetime and three-step
density directional derivatives. The benchmark's `--spatial` mode exercises
96 E/V combinations: four saved fixtures, LDA/PBE, three spin/layout choices,
two budgets and screening off/on. The screened Python oracle zeroes omitted
global AO columns; its difference from the unscreened saved fixture is recorded
separately from D/C arithmetic error. Five interleaved pairs per case include
all packing, transfers and CPU assembly. These small inputs do not establish
a performance winner or a complete molecular endpoint.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VIBEQC_NVCC=/group/software/cuda-12.9.1/bin/nvcc \
  .venv/bin/python tools/benchmark_density_sources.py --spatial \
  --library build/cpu/libvibeqc.so --cache .artifacts/spatial-density-bench-cache \
  --output .artifacts/benchmarks/spatial-density-sources --samples 5
```
