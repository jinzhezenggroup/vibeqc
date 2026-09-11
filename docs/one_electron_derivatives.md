# Generated one-electron derivatives

VibeQC provides an opt-in generated CUDA consumer for first nuclear derivatives
of overlap S, kinetic T and nuclear attraction V. It accepts arbitrary external
weights, and the Direct/DF RHF/UHF adapters use that same contract. The
handwritten implementation remains the default.

```bash
VIBEQC_ONE_ELECTRON_DERIVATIVES=generated \
VIBEQC_ONE_ELECTRON_DERIVATIVE_MAPPING=thread your-command
```

Select `reference` to use the previous force implementation. The `thread`
mapping owns triangular AO pairs by thread; `shell_warp` assigns a shell pair
to a warp whose lanes own AO components. `serial` is a deterministic diagnostic
mapping with one owner per system. None of these switches changes the value
implementation selected by `VIBEQC_ONE_ELECTRON_VALUES`.

## Mathematical and weight contract

The generator differentiates the validated S/T/V value DAG before emission.
Hermite terms, Gaussian decay, prefactors and Boys values share graph nodes.
Boys leaves use the analytic rule `dF_n(T) = -F_(n+1)(T) dT`. The emitted
arithmetic contains scalar operations and a bounded Boys evaluator, with no
runtime automatic-differentiation objects. Public Cartesian and real-spherical
s/p/d/f conventions, shell radial normalization and sparse Cartesian expansion
coefficients are inherited from the existing basis layer.

S/T have mathematical Gaussian centers A/B, with `dB = -dA`. Attraction has a
distinct external nuclear center C, with `dC = -(dA+dB)`; neither the external
center nor its charge is silently folded into a Gaussian shell. A/B derivatives
remain independent even when their physical atom indices coincide. Physical
atom ownership is applied only during accumulation. Nuclear charges multiply
the generated signed unit-charge attraction. First derivatives require Boys
orders through seven for f/f. The public angular-momentum limit is unchanged.

The generic output is the energy gradient

```
g[R] = sum_ij (W_S[ij] dS[ij]/dR
             + W_T[ij] dT[ij]/dR
             + W_V[ij] dV[ij]/dR).
```

Each weight uses a full `[system, AO, AO]` matrix. Weights are fixed inputs to
the current Lagrangian derivative; this layer does not differentiate them.
Triangular ownership uses `W[ij]+W[ji]` off the diagonal and `W[ii]` once on the
diagonal. Consequently nonsymmetric and non-HF weights have the same full-dot
semantics. There is no implicit factor of two in the generic consumer.

The stationary HF adapter supplies density for T/V and negative energy-weighted
density for S. RHF matrices already contain their double-occupation factors.
UHF sums alpha and beta matrices before calling the same consumer. API forces
are the negative final energy gradient. Nuclear repulsion is computed by its
independent module and added exactly once.

## Native interfaces and storage

`launch_generated_one_electron_gradient` consumes non-owning device topology,
geometry, weights and an optional active-item mask on the caller's owning
stream. It adds into an existing `3*Natom` buffer, so a caller can first place
the nuclear contribution there. This launch performs no allocation, transfer
or synchronization. Pair-center accumulators occupy registers; external nuclear
responses go directly to the output. Different atoms and derivative axes are
handled within each primitive/component traversal, without regenerating all
integrals separately for every global nuclear coordinate.

The public C function `vibeqc_system_one_electron_gradient_cuda` provides a
synchronous standalone bridge. Its three weight pointers use row-major full AO
matrices; a null channel means zero. `matrix_count` must equal `NAO*NAO`, and
`gradient_count` must equal `3*Natom`. A CUDA context is required. CPU-only
builds return `NOT_IMPLEMENTED`; generated runtime failures are returned to the
caller instead of silently selecting a CPU implementation.

The bridge stages compact basis metadata and weights on the device and returns
only `3*Natom` gradient scalars. A finite `maximum_bytes` bounds its numeric host
staging and device arena separately. It checks a conservative capacity bound
before building metadata/pair vectors and checks every device allocation.
Caller-owned weights/system data, existing HF plans, allocator bookkeeping and
opaque driver/library allocations are outside this bound. The optional
`vibeqc_one_electron_gradient_resources` descriptor reports numeric capacities,
H2D/D2H bytes, synchronous upload calls and explicit stream synchronization
calls. These counters do not claim to expose synchronization internal to CUDA
allocation or driver APIs.

Prepared Direct HF uses its existing device density, energy-weighted density,
geometry and force buffer. Its derivative selector is read on each force
execution and requires no candidate-specific cached geometry or buffers.
Failed-item masks prevent its force kernel from consuming an unconverged
neighbor's density.

Standalone DF retains the public basis/geometry for its final response and
omits the full `3*Natom*NAO*NAO` S/Hcore derivative arrays when the generated
consumer is selected. It still obtains one-electron values and the independent
O(Natom) nuclear-repulsion response. Cached DF data compare implementation and
mapping policy before reuse. The existing DF J/K and two-electron response
paths keep their own placement; the new one-electron bridge uploads final
weights and returns a small gradient for final assembly. This host boundary is
explicit, and this change does not claim a fully device-resident DF workflow.

## Validation and reproduction

The CPU symbolic/emitted suite checks all public shell pairs against libcint
and finite differences, including independent center signs, arbitrary weights,
translation, coincident centers, normalized contractions and spherical output.
The raw CUDA protocol uses 83 fixtures and 11,497 primitive records, covering
asymmetric, coincident/near-coincident, intermediate/distant and long-contraction
cases. Raw first-derivative diagnostics retain their explicit center and axis
dimensions; they do not become production tensor allocations.

```bash
PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest tests/python/test_one_electron_derivatives.py -q
PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python tools/validate_one_electron_values.py --derivatives \
  --directory derivative-raw --nvcc /path/to/nvcc
```

The raw runner compiles locally and obtains finite Slurm GPU allocations for
execution. `--local` is accepted only within an existing allocation. Enabled
endpoint tests similarly require Slurm and fail on CUDA errors rather than
converting them into skips:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:15:00 bash -lc 'PYTHONPATH=python:. \
  VIBEQC_LIBRARY=$PWD/build-cuda/libvibeqc.so \
  VIBEQC_ONE_ELECTRON_DERIVATIVE_CUDA_TEST=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest tests/python/test_one_electron_derivatives_cuda.py -q'
```

The GPU tests compare arbitrary nonsymmetric-weight raw dot products with all
three fused schedules, check deterministic replay and bounded output transfers,
and exercise Direct/DF RHF/UHF in Cartesian/spherical representations on cold,
warm and changed geometries. They also check selector transitions on reused
plans, independent analytic target forces, rigid transformations and failed
neighbors. One-electron contractions are unscreened; ERI screening remains the
responsibility of the existing two-electron force path.

`benchmarks/one_electron_values_gate.py --derivatives` retains five interleaved
baseline/candidate pairs for complete cold, unchanged and changed-geometry
energy-plus-force endpoints. `--fitted` compares DF's previous full one-electron
derivative tensors with the fused path, and `--df-budget` controls the existing
DF workspace. `--contraction-length` builds longer explicit contractions for
custom-shell cases. `--observe-resources` retains resource-plan observations.
All paired energies/forces are checked; timings are whole endpoints, not
isolated integral-kernel claims. Reduced raw arithmetic cost alone cannot
promote a production force path.

## RTX 5090 evidence and disposition

The [archived acceptance record](../benchmarks/results/one-electron-derivatives-rtx5090/validation-summary.json)
pins source, binary, compiler and Slurm identities. All 56 symbolic/emitted CPU
tests and 13 native tests passed. Enabled GPU tiers passed 25 derivative tests,
four screening/finite-difference tests, 41 checkpoint tests and 14 resource
tests, with no skips. Independent raw CUDA blocks have maximum absolute error
`6.67e-14`; arbitrary-weight full-matrix contractions agree within `1.07e-13`.
The complete-force screening tests use `1e-30` as a near-unscreened reference
within the positive-threshold API, then check `1e-14` and `1e-9` against it and
against energy differences at two step sizes. These are fixture-specific error
measurements, not a universal bound implied by the screening threshold.

The initial full-axis lowering is retained under `initial-full-axis/`. Axis
permutation reduced emitted header size from 16.69 to 7.65 MB, object size from
11.14 to 4.29 MB and probe compilation from 171.1 to 51.6 seconds. Worst helper
spill stores/loads fell from 66,688/69,528 to 6,488/6,516 bytes. Raw primitive
runtime did not improve: approximately 1.028/1.113 ms for 64/128 threads versus
1.006/1.065 ms initially. The lowering was retained for its smaller compiler
workload and storage demand. Native fused kernels still use 255 registers and
1,152–1,168 bytes of stack; they do not establish a spill-free implementation.

Whole energy-plus-force endpoint medians below are milliseconds, each from five
interleaved pairs. Each cell shows reference / generated. All paired energy and
force gates pass; maximum paired force error is below `7.6e-15`.

| Workload | Cold | Unchanged geometry | Changed geometry |
| --- | ---: | ---: | ---: |
| Direct sp8, batch 3, thread | 55.655 / 55.420 | 3.241 / 3.210 | 13.420 / 13.745 |
| Direct sp8, batch 3, shell warp | 54.980 / 54.844 | 3.198 / 3.155 | 13.450 / 13.590 |
| Direct sdf18, thread | 46.781 / 47.262 | 16.017 / 16.388 | 36.667 / 39.352 |
| Direct sp8, eight primitives/shell | 2014.792 / 2005.749 | 1472.445 / 1482.119 | 10015.361 / 10036.744 |
| DF sp8, batch 3, resident | 63.317 / 61.280 | 4.586 / 5.429 | 14.568 / 13.900 |
| DF sp8, batch 3, 1 MiB source budget | 212.792 / 210.675 | 110.901 / 108.709 | 138.419 / 137.468 |

The `profile-*.nsys-rep` traces and CSV summaries reuse
`benchmarks/profile_one_electron_force.py`. On sp8 batch 3, mean one-electron
kernel times are 335.6 us for the previous scalar kernel, 85.2 us for its
cooperative kernel, 121.8 us for generated AO threads and 58.0 us for generated
shell warps. These separately profiled kernel numbers do not replace the
interleaved endpoint gate. Direct generated execution adds no allocation or
transfer. DF profiling confirms its remaining host boundary: five replays of
three systems add 15 gradient downloads and explicit stream synchronizations,
plus metadata and weight uploads.

The standalone arbitrary-weight reports `contract-*.json` record actual numeric
staging capacities and transfers. On sdf18, the explicit raw dS/dT/dV oracle
contains 46,656 bytes, whereas the shell-warp bridge uses 1,320 host numeric
bytes and an 8,820-byte device arena, returning 48 bytes. The previous DF S/H
derivative pair alone contains 31,104 bytes per system and is omitted by the
generated path. These tensor/staging figures exclude caller inputs and opaque
CUDA storage. The observed whole-HF owned-device peaks remain equal between
reference and generated: 146,896 bytes for Direct sp8 batch 3, 628,324 for
resident DF and 617,130 for source DF. Other phases dominate those peaks;
complete process host high-water is not claimed. Resource-budgeted runs omit
optional iteration-history profiling, while retaining final residuals and
iteration counts.

The generated implementation remains **opt-in**: gains are workload-dependent,
and unchanged resident DF replay regresses. Reproduction scripts are archived
alongside the results and resolve the checkout relative to their own location.
Set `PYTHON`, optional `CUDA_HOME`/`NSYS`, and run each through a finite Slurm
allocation. `VIBEQC_LIBRARY` can select a nondefault build.

Integration with the merged basis-projection implementation and review fixes
is recorded separately in `integration/validation.json` beside the original
evidence. It passes 13 native CUDA tests, 29 derivative tests, 5 projection
tests, 41 checkpoint tests and 14 resource tests. The RHF/UHF cache regression
rejects an infeasible replacement budget and verifies recovery at 1/8 MiB,
rebuilding both prepared response data and the owning CUDA plan. The bridge
restores the calling thread's device after cleanup; this single-GPU validation
does not exercise restoration between distinct physical devices. Original
benchmark hashes and timings above remain attached to the original binary.
