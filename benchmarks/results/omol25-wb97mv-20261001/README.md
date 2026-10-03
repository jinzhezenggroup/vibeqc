# OMol25-level ωB97M-V/def2-TZVPD versus GPU4PySCF

![Current automatic warm energy and analytic-force endpoints](default-hf-cartesian/omol25.svg)

## Preserved automatic Cartesian-source endpoint

The current curve uses the automatically selected HF Cartesian source,
shell-local spherical projections and screened canonical traversal. Slurm
11965 completes all twelve independent energy/force gates at 3 atoms/58 AOs:
original warm **6.300998 s**, moved warm **6.308965 s**, cold **30.716030 s**
and moved **21.447856 s**. GPU4PySCF's corresponding warm medians are
**16.424809 s** and **16.471310 s**. Maximum absolute energy and force errors
are `8.27e-12 Eh` and `5.79e-11 Eh/Bohr`.

[Current scalar evidence and provenance](default-hf-cartesian/water3.json.gz)
and compressed complete native/reference journals preserve the original
`737f3481fcc5a82039bd59b676f14d5e963a517d`-based candidate, library hash and
source-file identities; they are not relabeled as measurements of a later
master or the PBE0 benchmark binary. Native library SHA-256 is
`c693f5fc95e0ad4c16988fcade63da31ecfc7e5ebc3b990da39dbc78de5e935d`.

The six-atom native process (Slurm 11966) times out at 540 s during the moved
endpoint. Its partial original warm observations (~52.4 s) are retained in
[the journal](default-hf-cartesian/water6-native.json.gz), but **do not qualify
as a complete-protocol performance point**. Its independent reference is
complete. Larger native points remain unmeasured; the unchanged 1024-AO force
cap still excludes 96 atoms/1856 AOs. No through-100-atom speed claim follows
from this small-system result.

The earlier [opt-in preview curve](omol25.svg), JSON and qualification
records remain intact. The historical snapshot below records the earlier
schedule and baseline; its opt-in settings and old source revision do not
describe the current production default.

## Historical preview snapshot

This benchmark uses **OMol25's functional and orbital basis**, not its molecular
distribution or ORCA implementation. The level of theory is documented in the
[OMol25 paper](https://arxiv.org/abs/2505.08762). It keeps the HF README's nested
water32mer prefixes, complete energy-plus-analytic-force endpoint, fixed
engine-local density seeds, and five warm repeats. Spherical def2-TZVPD includes
all diffuse shells, yielding 58/116/232/464/928/1856 AOs at 3/6/12/24/48/96 atoms.
Density fitting is outside the native WB97M-V force contract, so both plotted
engines use direct integrals. The HF figure is unchanged.

**The through-f public endpoint is repaired, not bypassed.** PR #1637 supplies
the geometry-only force composition; this change aligns public capability
discovery with that domain and fixes its nuclear-only dispatch to use plain
task kinds rather than Cartesian integral-kind encodings. Named def2-TZVP RKS
water, full local def2-TZVPD RKS water, and named def2-TZVP UKS NH₂ all pass
independent complete energy/analytic-force comparisons, translation, warm/cold
consistency and six reconverged displaced energies at three finite-difference
steps. [force-qualification.json](force-qualification.json) records accepted
cases and the enclosing job outcomes separately.

The basis is not truncated, the functional is not substituted, the native
production path does not borrow reference densities or derivatives, and
energy-only execution is not passed off as a force endpoint. The retained
bounded native one-electron metadata staging fallback is explicitly charged;
the final-state density/Pulay export gates remain zero. The 1024-AO force-owner
limit is unchanged, so 96 atoms/1856 AOs are outside that resource domain.

## Complete-endpoint measurements

**Speed-first snapshot:** the native curve uses the explicitly enabled
canonical-J/K PR-head preview (Slurm 11950), not the slow ordered-AO schedule.
The current source integrates GitHub master
`8154ab3df56a10900dd267039b857021a1054721` plus #1637 and the endpoint fixes;
the earlier preview timing is not relabeled as a measurement of that new build.
[endpoint-preview.json](endpoint-preview.json.gz) retains all twelve scalar calls
for the old native schedule, canonical preview and independent reference, with
their exact binary/source/raw identities.

| 3 atoms / 58 AOs, seconds | Native ordered-AO baseline | Native canonical preview | GPU4PySCF |
| --- | ---: | ---: | ---: |
| Cold complete endpoint | 879.220 | 64.289 | 111.356 |
| Original warm median, all five | 46.255 | 14.383 | 16.425 |
| Changed geometry, including refresh | 400.485 | 45.682 | 113.772 |
| Moved warm median, all five | 46.549 | 14.389 | 16.471 |

The original warm endpoint improves **3.22×**, cold **13.68×**, and changed
geometry **8.77×** against the recorded native baseline, without changing the
basis, grid, convergence or gates. All twelve preview calls pass the independent
gates; maximum errors are `7.92e-12 Eh` and `5.75e-11 Eh/Bohr`. This is one
qualified small-molecule point, not large-system parity. Six atoms currently
have a complete independent reference only; larger points are deliberately
unmeasured while performance work takes priority. They are not plotted as
native timings or timeout-derived bounds. The slow full matrix is stopped.

The historical preview runner explicitly selected
`GENERATIVEQC_CUDA_CANONICAL_JK=1`; the provider default was then unchanged
pending larger endpoint and candidate UKS-force qualification. The current
runner no longer needs this opt-in, and the old switch is not its production
performance contract.
See the [schedule and fallbacks](../../../.agents/notes/implemented/performance/2026-10-01-canonical-public-ao-jk.md).

The integrated master build passes short independent full/SR/LR matrix gates
through f, including both spins/AO representations, nonsymmetric densities,
output masks, two geometries and bounded fallback. Its instrumented scheduler
visits/evaluates exactly 1,464,616 and 23,028,291 unique ERIs per radial pass at
58 and 116 AOs. [schedule-qualification.json](schedule-qualification.json)
retains the current binary/source identity and Slurm 11955 evidence.
This synthetic single-primitive work census is **not** a molecular energy/force
endpoint benchmark and supplies no additional points for the performance curve.

Only a point with both geometries and all five original/moved warm replays
passing their independent gates qualifies for the plotted median. Every
failure and timeout remains a separate status, not a timing bound or an
extrapolated curve point.

## Protocol and scope

- Native source: GitHub `master` fetched on October 1, 2026,
  `8154ab3df56a10900dd267039b857021a1054721`, with #1637's geometry-only force
  support and the local public-dispatch/canonical-J/K fixes. The library uses Release
  CUDA 12.9.1 / sm_120 build with generated shell AOT enabled, stationary-force
  AOT disabled, and the unchanged default resource planners. Benchmark scripts
  are added on top of this revision; source status and binary hash accompany
  each result. Historical numerical qualification records PR head
  `6795097b003a30f9f5e9ab8c34e0705dddbf2427`; the implementation changes now
  live in Git history and PR #1651 rather than duplicated reconstruction files.
  Per-file hashes identify both the dirty Python consumers and native schedule;
  a native binary digest alone is insufficient.
- Both reference and native measurements are fresh matched-domain runs. The old
  GPU4PySCF default-domain reference timings are excluded, not reused. Each
  engine's source, environment and raw hash is retained without rewriting identity.
- RTX 5090 through Slurm `main`, `--gres=gpu:5090:1`; scheduler-assigned device
  visibility is preserved. OpenMP, OpenBLAS and MKL each use eight threads.
- PySCF 2.14.0, GPU4PySCF 1.8.1, NumPy 2.4.6, CuPy 13.6.0 and cuTENSOR 2.2.0.
  GPU4PySCF's independent SCF, AO/integral/XC/VV10 evaluation and analytic
  gradients are not replaced by a native/reference adapter. The shared atomic
  quadrature is the existing WB97M-V benchmark adapter; GPU4PySCF independently
  constructs molecular partitioning and its analytic grid response.
- The checked-in [canonical basis](def2-tzvpd-ho.json) and
  [MolSSI BSE source](def2-tzvpd-ho.bse.json) are an offline H/O snapshot from
  Basis Set Exchange 0.12, def2-TZVPD version 1. Both engines receive this exact
  data, including every general-contraction column and diffuse shell. No basis
  service is consulted during a timed run or native preparation.
- The common moving `GridSpec` has 48 radial points × 16 polar × 32 azimuthal
  points per atom, original-Becke partitioning, three partition iterations, no
  radius adjustment and no pruning. Semilocal XC, self-consistent VV10 and
  analytic partition/grid response are included. This matches both engines'
  quadrature; **it does not reproduce ORCA's OMol25 integration grid or establish
  basis/grid convergence for the dataset**.
- VV10 uses `rho >= 1e-8` on both pair legs, matching native MolecularV1 and
  PySCF CPU. GPU4PySCF 1.8.1 defaults to `1e-10` and copies its threshold into
  the force module at import; this comparator explicitly scopes **both** masks
  together and restores them on exit. Its independent kernels and analytic
  response are unchanged. Its separate `|weight| > 1e-14` screening is recorded.
  Libxc is 7.0.0 in the CPU and CUDA reference. Sharing grid coordinates alone
  is not sufficient to match VV10 work.
- Native SCF uses energy tolerance `1e-12 Eh`, density tolerance `1e-10`,
  screening `1e-12`, and at most 100 iterations. GPU4PySCF uses energy tolerance
  `1e-12 Eh`, orbital-gradient tolerance `1e-10`, direct screening `1e-14`, and
  at most 100 cycles. These match the HF benchmark's public tolerances, not its
  internal convergence algorithm or electronic work counts.
- Independent reference and native engines run in separate sequential processes.
  Cold includes preparation and the first synchronized full endpoint, but
  excludes imports, calculator/library loading and metadata probes. It does
  not imply an empty filesystem/compiler cache. All five
  warm calls reuse the engine's fixed post-cold converged density. The second
  atom then moves +0.001 Bohr along z, reconverges, and supplies the frozen seed
  for five moved-warm calls. Geometry refresh is included in the moved endpoint.
  Both clocks include returning the force array to host memory, including
  GPU4PySCF's `cp.asnumpy` transfer before the clock stops, exactly as in HF.
  Schema v3 rejects the earlier kernel-complete/device-force timing records;
  neither those reference latencies nor their paired native records are reused.
- Every completed native cold/warm/moved/moved-warm call is rechecked against
  the independent matching-geometry reference: `|ΔE| <= 1e-8 Eh` and
  `max|ΔF| <= 1e-7 Eh/Bohr`, with explicit shape, finite-value and convergence
  gates. Every reference replay is rechecked against its own cold/moved oracle.
  One bad call rejects the point; matching SCF iterations never selects samples.
  Native capability preflight occurs before SCF, and every measured native call
  explicitly requests both energy and forces. Partial success never qualifies
  a full point for the plotted curve.
- Curves use medians of **all five** warm endpoint latencies, with min/max bars;
  crosses expose variable-iteration repeats without removing them from medians.
  Times are not divided by iterations. Reference `get_veff` work counts include
  the pre-loop Fock; native public Fock counts remain `null` when unavailable.
  Native derivative component/work metadata is retained separately, without
  claiming a component-time split.
- Each engine/size process has a finite whole-point limit, including
  cold and changed-geometry qualification. The complete canonical preview has a
  330-second limit; the prior slow native baseline had a 5400-second limit.
  Limits/outcomes are retained with each raw point. A timeout is **not** a warm endpoint
  timing or a lower bound on warm latency. Reference evidence survives native
  failure. Unmeasured or failed points are explicitly identified in the figure
  and retained JSON rather than extrapolated.

## Evidence

Each `water<N>.json` retains the exact scientific protocol, coordinates, basis
identity, independent original/moved energy-and-force oracles, all scalar
endpoint observations, convergence and semantic work, binary/source/environment
identity, scheduler IDs, process outcomes, and hashes of the ignored raw files.
The reducer rechecks every full raw endpoint before compacting force arrays.
Logs, full raw repetitions and binaries remain in ignored local artifacts.

[public-api-probe.json](public-api-probe.json) retains **historical, unmodified
PR** explicit public force
requests for local def2-TZVPD and the PR's named def2-TZVP case, with def2-SVP as
a supported-capability control. Those native capability probes ran under Slurm 11925;
the independent explicit public requests ran under Slurm 11926. The unmodified
PR's layout admission was insufficient: public force requests were rejected
before SCF. This is superseded diagnosis, not the current native measurement.

Initial CuPy 14.2 fallback-einsum environment probes are excluded. All retained
measurements use the pinned CuPy/cuTENSOR environment above.

An initial native call without capability preflight completed energy-only work
on the base master in 602.84 s and returned no forces despite a successful
status. Its complete-force gate rejected the call. This excluded diagnostic is
**not a force latency**. The earlier default-domain GPU4PySCF medians are also
excluded: both engines are rerun with the explicit matched VV10 domain.

The initial OH-doublet UKS acceptance fixture was rejected because its
independent GPU4PySCF SCF did not converge, including 400-cycle, level-shifted
and finer-grid probes. The NH₂ replacement uses the original small grid,
original stopping criteria, the same through-f basis, and unchanged independent
force/finite-difference gates. No OH oracle or accepted timing is claimed.

The focused host suite passes **110 tests, 16 opt-in skips**. A broader host
run has one CPU-only water finite-difference SCF nonconvergence; it reproduces
on archived **unmodified PR Python sources** with the same native library and
eight threads. That unrelated convergence behavior is not changed here.

## Reproduce

Use the repository Python dependencies plus the versions above; install
`basis-set-exchange==0.12` only to regenerate the optional offline source. The
retained basis inputs themselves need no BSE package or network service.
Use the benchmark/comparator scripts from this change. Historical timing
records retain their source hashes and commit/PR identities; the implementation
now lives in Git history rather than duplicated reconstruction files. For new
runs, rebuild the current implementation and start with a bounded three-atom
point rather than waiting for the full slow matrix.

```bash
cmake --preset cuda-release-sm120 \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=OFF
cmake --build --preset cuda-release-sm120 --target generativeqc -j10
export CUDA_PATH=/group/software/cuda-12.9.1
export PATH="$CUDA_PATH/bin:$PATH"
cutensor_lib=$(python -c 'from pathlib import Path; import cutensor; print(Path(cutensor.__path__[0]) / "lib")')
export LD_LIBRARY_PATH="$cutensor_lib:$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
export OMOL25_BASIS_FILE="$PWD/benchmarks/results/omol25-wb97mv-20261001/def2-tzvpd-ho.json"
export OMOL25_BENCHMARK_PYTHON="$(command -v python)"
export OMOL25_BENCHMARK_OUTPUT="$PWD/.artifacts/omol25-reproduction"
export GENERATIVEQC_CUDA_CANONICAL_JK=1
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:15:00 bash -lc '
    OMOL25_ATOMS=3 OMOL25_POINT_TIMEOUT=420 bash benchmarks/run_omol25_benchmarks.sh
  '
```

The runner returns nonzero if any point fails or times out, but continues the
matrix and preserves all outcomes. Native unsupported endpoints also return
nonzero. Run the reducer even after a nonzero exit:

```bash
PYTHONPATH=python:. python -m tools.render_omol25_benchmarks \
  --raw-directory "$OMOL25_BENCHMARK_OUTPUT" \
  --basis-directory benchmarks/results/omol25-wb97mv-20261001 \
  --destination .artifacts/omol25-reviewed
PYTHONPATH=python:. python -m pytest -q tests/python/test_omol25_benchmark.py
```

`OMOL25_ATOMS`, `OMOL25_ENGINES`, `OMOL25_REPEATS` and `OMOL25_POINT_TIMEOUT`
permit bounded diagnostic subsets; a changed repeat count or timeout is recorded
and must not be silently presented as this five-repeat protocol.

To reproduce the current through-f numerical qualification:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=02:30:00 \
  env PYTHONPATH=python:. GENERATIVEQC_TEST_WB97MV_CUDA=1 \
  python -m pytest -q tests/python/test_wb97mv_complete_cuda.py \
  -k 'test_complete_cuda_force_matches_independent_engine and (def2-tzvp or local-def2-tzvpd)'
```

## Lossless report storage

Five earlier scalar reports now use `.json.gz`; [report-storage.json](report-storage.json)
records decoded/stored sizes and SHA-256 digests. `gzip -dc FILE.json.gz` recovers
every original byte. Basis inputs, native/reference journals and all scientific
observations are unchanged. The renderer consumes original run journals and
emits these scalar reports; no executable consumer of these five stored reports
was found in the source/test audit.
