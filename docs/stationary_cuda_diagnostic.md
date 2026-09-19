# Stationary CUDA RKS gradient diagnostic

`vibeqc._stationary_cuda.complete_rks_cuda_gradient_diagnostic` executes all
`StationaryGradientPlan` sources on CUDA: one-electron, Coulomb, XC AO motion,
XC point motion, XC partition response, overlap/Pulay, and nuclear repulsion.
The result is an energy gradient in Eh/bohr; force is its negative. The public
Python C2 endpoint reuses the all-electron subset for qualified CUDA LDA/PBE
RKS/UKS forces. The ECP extension here remains diagnostic-only: ECP basis sets
are energy-only at the public `Calculator` boundary, and its force wrapper
independently rejects an ECP snapshot before compilation or contraction.
This diagnostic does not close #163 or #396 or qualify public ECP forces.

The admitted domain is direct, all-electron, real FP64 integer RKS/UKS with canonical
LDA/PBE, s/p single-component AOs and the native unpruned version-one grid.
Distinct nuclei and no point/center collisions are required, including at zero
weights. Density fitting, hybrids, meta-GGAs, higher angular momentum,
Hessians and public production resource qualification are outside this contract.
Ordinary stationary first derivatives require no CPKS/Hessian solve.

## Execution and ownership

CUDA snapshot wire v3 appends the actual owner's GridSpec, raw atomic measures
and measured snapshot-export D2H/read/synchronization counters. CPU wire v2 is
unchanged. Legacy CUDA v1 snapshots remain readable, but cannot enter this
complete diagnostic. Neither Python labels nor copied matrices can manufacture
the live opaque native token. Replay, closure, replacement and rejected updates
revoke the old snapshot; the token is checked again before publication.

The complete route retains these explicit host boundaries:

- Native CUDA SCF constructs its grid and has existing provider setup boundaries.
  The final D/F/C/epsilon export, validation and W construction are explicit host
  operations. The derivative function consumes that exported snapshot.
- Python enumerates ordered AO pairs/quartets and primitive records, gathers
  density entries, and packs exponents, centers and normalization coefficients.
  It does not evaluate a derivative or reduce scientific contributions.
- `plan_cuda`/`compile_cuda`/`PreparedCuda` execute source weights and the final
  complete-source reduction. Their inputs and small outputs stage through the host.
- Existing `CudaGrid` evaluates AO jets and density features. The geometry
  consumer borrows AO jets, D-contracted jets, features and the producer stream
  during the locked task lease; those arrays never download for differentiation.
- Generated integral graphs, the existing SCF point model
  `semilocal-scaled-v1/pbe-spin-c2-1e-18`, generated AO bilinear pullbacks and the
  shared CPU/CUDA Becke adjoint execute on device. No interior diagnostic XC
  model, CPU derivative/contraction or interpreter fallback is selected.

Native code owns traversal, primitive normalization, atom scatter and bounded
reductions. The Becke worker shares one two-pass implementation across backends,
including the single-zero-factor derivative and saturated-branch policy.
Primitive CPU emitted bytes are preserved. Device compilation disables FMA
contraction; no broad fast-math flag or relaxed acceptance threshold is used.

## Bounded resources and failure

Preparation admits at most 32 atoms, 128 AOs, 4096 points per tile, 4096 primitive
records per tile, and 128 source-weight terms. Defaults cap total primitive work
at 2,000,000 records, grid points at 1,000,000 and grid pair visits at 100,000,000.
For `A` atoms, `N` AOs, point capacity `P` and primitive capacity `R`, the new
source arena owns exactly `8*(42*R + 600*A + 3*P + N) + 256` bytes. The Becke
scratch has 32 atom-sized worker slices; there is no coordinate/grid/AO tensor.
Ordered primitive work is `K**4 + (A+2)*K**2 + A*(A-1)/2`, where `K` sums primitive
counts over public AOs. Pair visits are `(1+2*grid_points)*A*(A-1)/2`.

All TensorIR programs and the grid/source capacities are admitted before device
allocation. Default additional-device and host-numeric bounds are 512 MiB and
256 MiB. The device bound conservatively also charges adapter host capacities.
The caller's existing SCF owner/snapshot, Python object headers, compiler
processes/graphs, mapped code and CUDA context/module/stack overhead remain
explicit exclusions. This is not a global SCF-plus-derivative reservation.

The source arena has one private stream and retains no borrowed grid pointers.
Geometry work finishes on the grid owner's stream before releasing its lease,
including exceptional exits. Device ordinal comes from the actual snapshot and
must match the current CUDA device and borrowed owner. No visibility override is
used. A native failure poisons the source transaction; reads fail until reset.
The public diagnostic discards the owner and publishes no partial result.

`result.work` records exact source launches, primitive/point/pair counts,
source H2D/D2H, snapshot export counters, streams, grid allocation/timing metrics,
TensorIR execution/transfer totals, declared numeric bounds, endpoint time and
the paths/hashes of every loaded generated artifact. The reused grid/TensorIR
ABIs do not expose a complete endpoint kernel-launch count; source launches
must not be presented as the endpoint total. No speedup is claimed.

## Example and qualification

Run GPU commands through the existing Slurm execution profile. For this machine,
build with the full CUDA 12.9 toolkit and `sm_120`; the launcher selects
`main`/`gpu:5090:1`, a finite time, and preserves Slurm device visibility.

```python
from pathlib import Path
from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc._dft_gradient import StationaryKsState
from vibeqc._stationary_cuda import complete_rks_cuda_gradient_diagnostic
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.dft import NativeAO

atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
compiler = CudaCompilerAdapter(Path("nvcc"),
                               cuda_target_info("sm_120"), compile_timeout=600)
calc = Calculator(method="pbe-rks", device="cuda",
                  ks_options=KsOptions(grid=GridSpec(radial_points=24,
                      angular_polar=8, angular_azimuth=16)),
                  energy_tolerance=1e-12, density_tolerance=1e-10)
with calc.prepare_batch([atoms]) as batch, NativeAO(atoms) as basis:
    energy = batch.execute(strict=True).items[0].energy
    state = StationaryKsState.from_native(batch, basis)
    result = complete_rks_cuda_gradient_diagnostic(
        state, basis, compiler=compiler, cache=".cache/stationary-cuda")
    forces = -result.gradient
```

From the repository root, using a Python with NumPy, pytest and PySCF:

```sh
python tools/run_stationary_cuda_validation.py
python tools/run_stationary_cuda_validation.py --full-fd -k reconverged
python tools/run_stationary_cuda_validation.py --sanitizer memcheck -k 'analytic or source_failure'
python tools/run_stationary_cuda_validation.py --sanitizer initcheck -k 'analytic or source_failure'
python tools/run_stationary_cuda_validation.py --cpu-regression
```

The opt-in device suite compares H2 and asymmetric s/p water, LDA and PBE,
against independent PySCF/Libcint/Libxc analytic gradients including full grid
response. Every signed source has a `1e-7` Eh/bohr gate. Multistep reconverged
finite differences (all Cartesian coordinates at three steps with `--full-fd`)
use a `1e-6` raw coordinate gate and `1e-7` extrapolated gate.
Failure tests cover malformed input, byte/work admission, corruption, invalid
device, stale/replaced state, late failure, recovery, empty tiles, exact-zero
products and vacuum tails. CPU scientific/interpreter entrypoints are blocked
during complete CUDA execution. Sanitizer runs are separately invoked/counted.
Evidence is retained locally in ignored `build-cuda/stationary-evidence/`.
`--library`, `--cache` and `--evidence` select explicit local paths; the existing
Slurm profile environment selects partition and GPU request. CPU regression
runs locally without reserving a GPU.

The [decision record](../.agents/notes/implemented/architecture/2026-09-19-stationary-cuda-diagnostic.md)
preserves shared-science choices, measured evidence and remaining qualification.

## Strict compilation environment

The stationary CUDA compiler rejects nonempty `NVCC_PREPEND_FLAGS` and
`NVCC_APPEND_FLAGS` before generation or cache publication. External flags
cannot silently override the qualified FP64 arithmetic policy. Recording an
override in artifact metadata is not numerical qualification. Use the explicit
compiler adapter for supported target/toolchain selection.

## Scalar ECP diagnostic

CUDA snapshot v5 extends v3 with the live energy owner's exact core counts and
Gaussian ECP records. CPU v4 and all-electron CPU v2 / CUDA v3 layouts remain
unchanged. ECP parameters enter the model identity; replay and closure revoke
the derivative lease. Legacy CUDA snapshots without bound ECP parameters still
reject core-adjusted occupations.

The scalar ECP route uses effective ionic charges for attraction and nuclear
repulsion and adds `ecp_local` / `ecp_nonlocal` to the complete nine-source plan.
The existing generated CUDA ECP provider applies its two-grid convergence gate
and exports both all-center derivative arrays. TensorIR contracts the full
ordered AO pairs with spin-summed density and reduces all nine sources on CUDA.
No CPU ECP derivative or contraction fallback exists. The independent CPU ECP
provider remains available for validation.

This is an explicit dense host export, capped at 16 AOs, 8 atoms, 128 primitives
and 128 ECP terms. Qualification covers Cartesian s/p LANL2DZ Na / STO-3G H for
LDA/PBE RKS/UKS; these caps do not qualify arbitrary elements or parameter sets.
The provider runs before the other CUDA gradient owners are allocated; its
conservative two-grid workspace is admitted separately. Host bounds include the
re-read final-state snapshot, provider workspace and dense derivative copies.
The ECP workspace and export size are reported separately in `result.work`;
ordinary source launch/primitive counters do not include ECP-provider kernels.
The provider re-reads and validates the native final state, causing an additional
explicit final-state export; this is not an entirely resident force path or a
performance promotion. Public DFT/ECP forces remain gated.

Run the opt-in numerical gate on an allocated device with `VIBEQC_ECP_CUDA_TEST=1`,
`VIBEQC_ECP_CUDA_TARGET` matching that device (for example `sm_89`), an explicit
`CUDACXX` and current `VIBEQC_LIBRARY`:

```sh
python -m pytest tests/python/test_ecp_stationary_cuda.py -q
```

Diagnostic tests explicitly request `properties=("energy",)` when preparing
snapshots, so the C2 default property set cannot trigger an unrelated public
force calculation. Host-side admission and live-owner wrapper regressions in
`tests/python/test_ecp_cuda_public_boundary.py` run without a CUDA device; they
do not replace this real-device numerical gate.

The gate compares complete gradients with independent PySCF full-grid-response
analytic gradients and three-step reconverged energy differences. It also checks
raw derivatives against the CPU oracle, translation, ionic nuclear charges,
replay/closure/model identity, admission, late failure recovery and all-electron
v3 regression. See the [decision note](../.agents/notes/implemented/numerics/2026-09-19-cuda-ecp-stationary.md).
