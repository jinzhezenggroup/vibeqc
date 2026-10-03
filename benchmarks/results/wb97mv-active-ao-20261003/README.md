# Matched ωB97M-V energy and analytic-force endpoints

The explicitly enabled joint SCF/force active-AO integration has lower complete
warm latency than its paired GPU4PySCF reference at 3, 6, 12, 24, 48 and 96 atoms.
This is a measured candidate snapshot, not a claim about the master default or either
engine's default quadrature. Same-binary dense controls complete at 24/48/96 atoms.

![Warm WB97M-V energy and analytic-force latency](wb97mv.svg)

| Atoms / spherical AOs | Native dense warm | Native joint warm | Joint-paired reference warm | Joint / reference |
| --- | ---: | ---: | ---: | ---: |
| 3 / 24 | — | 0.963060 s | 2.051325 s | 0.469482 |
| 6 / 48 | — | 2.324435 s | 3.531840 s | 0.658137 |
| 12 / 96 | — | 7.265597 s | 8.820812 s | 0.823688 |
| 24 / 192 | 27.110692 s | 26.330918 s | 27.397887 s | 0.961057 |
| 48 / 384 | 106.139498 s | 97.112717 s | 103.042334 s | 0.942455 |
| 96 / 768 | 457.225580 s | 387.179006 s | 403.542077 s | 0.959451 |

The six-point curve covers the same 24/48/96/192/384/768 AO sizes as HF.
The 3/6/12-atom points are fresh same-protocol measurements of the identical
0132/b82 integration; no older timing is spliced into this curve. Dense
controls were not part of the small-point campaign (shown as —).

All three warm samples enter each median. The dense variants have their own
interleaved reference samples; those medians are 27.402939, 103.253732 and
404.221890 s at 24/48/96 atoms.
Native joint 48-atom samples are 97.129844, 97.112717 and 97.014143 s; reference
samples are 103.055296, 103.042334 and 103.035995 s. Every warm/priming result
converges in one iteration, using its own engine's frozen post-cold density.
There is no cross-engine density injection or iteration-normalized timing.
The 96-atom native samples are 384.612767, 387.179006 and 387.251892 s;
paired reference samples are 403.542077, 403.323987 and 405.490350 s.
The completed 96-atom dense samples are 455.943325, 457.225580 and
457.635133 s. The joint-map median is 15.32% lower than its same-binary dense
control; both have one SCF iteration in every warm call. This isolates the
combined SCF/force map selection, not either map alone.

## Cold startup and work

Cold includes calculator/engine construction, synchronized preparation and the
first complete execution. Imports and separate library probes are outside this
boundary; this does not assert an empty compiler/filesystem cache.

| Atoms | Native dense complete cold | Native joint complete cold | Joint-paired reference complete cold |
| --- | ---: | ---: | ---: |
| 3 | — | 10.528216 s | 9.948562 s |
| 6 | — | 24.632938 s | 19.075823 s |
| 12 | — | 89.063883 s | 45.447961 s |
| 24 | 249.833558 s | 240.398506 s | 116.268821 s |
| 48 | 1113.882289 s | 962.211204 s | 474.695895 s |
| 96 | 5016.225258 s | 4092.520636 s | 1851.051863 s |

Cold remains slower, particularly for larger clusters. Native cold submits
15/18/24/18/21/25 XC evaluations at 3/6/12/24/48/96 atoms; warm submits one.
The 96-atom reference cold takes
16 iterations. SCF discovery is preparation work,
not a cost added again to warm calls. At 48 atoms it takes 0.649430 s, retains
4608 tiles (436 empty), has an active-AO sum of 621388 and reports 14194184
numeric peak host bytes. SCF point/AO-square work is 14.5202% of dense; force
work is 18.7396%. These counters describe contraction work, not a speedup model.
The force map has no warm rediscovery. Dense-disabled force cache counters
are absent in the original report and are not replaced with invented zeros.

At 96 atoms, discovery takes 1.984494 s and reports 56699912 numeric peak host
bytes; 9216 SCF tiles retain 1460424 AO columns in total, with 768 empty tiles.
SCF/force point-AO-square fractions are 5.43804% / 5.82487%. The force planner
selects 256-point tiles at this size (9216 tiles), compared with 1024 at 24/48.
Warm force discovery remains zero. Native complete-cold preparation takes
5.781870 s; the first execution alone takes 4086.738766 s.
The same-binary dense control takes 5016.225258 s complete cold versus its
own paired reference 1849.279613 s. Native dense/joint cold trajectories take
23/25 iterations, so the cold reduction includes different convergence work
and is not attributed entirely to cheaper contractions. Both remain slower
than the reference.

## Protocol and numerical gates

- Water-cluster prefixes from `benchmarks.readme_hf_scaling.scaling_cases`,
  WB97M-V/RKS, complete spherical def2-SVP. Both engines use the same unpruned
  48 radial × 16 polar × 32 azimuthal grid per atom:
  73728/147456/294912/589824/1179648/2359296 points at 3/6/12/24/48/96 atoms.
  Semilocal and VV10 grids are the same full grid, with equal-radius Becke
  partitioning and analytic grid/weight response. Shared VV10 density threshold
  is 1e-8. This is not a comparison of the engines' respective default grids.
- Native energy/density/screening controls are 1e-11/1e-9/1e-12, maximum 180
  iterations. GPU4PySCF uses 1e-11 energy, 1e-8 orbital-gradient and 1e-14 direct
  screening tolerances, with full-density Fock builds. Different stopping
  criteria are retained, not presented as identical internal SCF algorithms.
- Explicit native SCF `GENERATIVEQC_CUDA_KS_ACTIVE_AO=1`; force caller uses
  `active_ao_cutoff=1e-16`, `active_ao_cache_bytes=64<<20`. Dense control disables
  both maps. SCF tiles remain 256 points; the composite force planner chooses
  its budgeted tiles. The fixed 64 MiB SCF host-map cap is experimental and
  does not establish public constrained-host-budget support.
- Slurm n1 job 5631 completes the fresh 3/6/12-atom candidate/reference
  measurements on one RTX 5090, verifying the original source, binary and all
  three measurement-script hashes before execution. n1 jobs 5577/5576 run each
  dense/joint comparison sequentially in one RTX 5090 allocation. n5 job 1412
  completes both 96-atom joint and dense controls sequentially. Each comparison preserves
  scheduler visibility. CUDA 12.9.1, CuPy
  13.6.0, NumPy 2.4.6, PySCF 2.14.0 and GPU4PySCF 1.8.1; eight OpenMP,
  OpenBLAS and MKL threads. Raw reports retain device and toolchain metadata.
- Cold, priming and all three warm pairs must each pass 1e-8 Eh / 1e-7
  Eh/Bohr, finite-value, shape and convergence checks. All 45 native/reference
  pairs across the nine completed variants pass. Maximum joint errors are below
  1.192e-10 Eh / 1.770e-9 Eh/Bohr. Every reference XC component reports
  `on_gpu=true`. No CPU XC fallback is used to explain the comparison.
- Independent complete WB97M-V qualification includes changed geometry,
  reconverged displaced energies and stale-snapshot isolation. Slurm 5575
  passes seven cases with joint maps and seven with force-cache allowance zero;
  each group verifies 66 successful native calls and 467 XC submissions.
  These are numerical qualification, not timed large changed-geometry results.

## Source, retained records and reproduction

Measured source is [0132d7584](https://github.com/jinzhezenggroup/generativeqc/commit/0132d75844c13d90bd04cb250e3aafae12d2a3dd),
an integration based on master d442177a6. Its 1351 build-manifest inputs match
the built source c06859af9. It includes the pending SCF/force AO stack and
qualified force improvements; it is broader than standalone PRs #1778/#1780.
The native/library identity and source archive verification are recorded in
[manifest.json](manifest.json), independently of the archived runner's unavailable
Git probe. Library SHA-256 is
`b82e468613c5c90e076aa10082b2389c8dda74335f625a7b77641c1c085ef3f9`.
Default screening is not promoted by this documentation.

The nine `.json.xz` files preserve every original report byte, including all
energy/force arrays, timings, convergence, preparation, XC backend and AO work.
The manifest records stored and decoded SHA-256 digests and sizes. Verify them
without a GPU:

```bash
python benchmarks/results/wb97mv-active-ao-20261003/verify.py 3
python benchmarks/results/wb97mv-active-ao-20261003/verify.py 6
python benchmarks/results/wb97mv-active-ao-20261003/verify.py 12
python benchmarks/results/wb97mv-active-ao-20261003/verify.py 24
python benchmarks/results/wb97mv-active-ao-20261003/verify.py 48
python benchmarks/results/wb97mv-active-ao-20261003/verify.py 96
```

Regenerate the README figure using the same plotting style as HF:

```bash
python -m tools.render_wb97mv_active_ao_benchmarks
```

The renderer verifies the retained endpoints and manifest hashes before drawing
the joint candidate and its paired reference. The README figure shows only warm
medians and min–max ranges on the same full AO axis and visual template as HF.
Complete cold totals stay in the detailed table above. Dense controls stay in the
detailed tables above.

Build the measured checkout in Release/sm_120 with CUDA 12.9.1 and explicit
CXX/CUDA `ccache` launchers after verifying `ccache --version`. Retain source
identity, compiler commands and pre/post cache statistics. No binary is hosted
in this evidence directory. The replay patch reconstructs the exact three
measurement scripts from the measured checkout's `benchmarks/readme_wb97mv.py`:

```bash
export ACTIVE_AO_ROOT=/data/jzzeng/wb97mv-replay
mkdir -p "$ACTIVE_AO_ROOT"
cp benchmarks/readme_wb97mv.py "$ACTIVE_AO_ROOT/complete-cold-benchmark.py"
patch -p1 -d "$ACTIVE_AO_ROOT" < /path/to/this/evidence/reproduce.patch
export PYTHONPATH="$PWD/python:$PWD"
export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export ACTIVE_AO_MODE=sparse GENERATIVEQC_CUDA_KS_ACTIVE_AO=1
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=01:00:00 \
  python "$ACTIVE_AO_ROOT/matched-probe.py" --atoms 48 --repeats 3 \
  --grid 48 16 32 --output "$ACTIVE_AO_ROOT/matched48-sparse.json"
```

Use `ACTIVE_AO_MODE=dense GENERATIVEQC_CUDA_KS_ACTIVE_AO=0` for the dense
control at the retained 24/48/96-atom sizes. For the fresh small candidate points,
keep sparse mode and use `--atoms 3`, `6` or `12`; all three measurement
scripts and scientific controls are byte-identical to the larger campaign.
For 96 atoms, allow `--time=04:00:00` for each variant. Supply the recorded CUDA runtime and
GPU4PySCF environment through the cluster's normal module/library setup.
Preserve Slurm's device visibility. The replay patch's source and reconstructed
script digests are retained; cold preparation is timed explicitly, and all
diagnostic hooks run after endpoint timers stop. Full build/scheduler/profiler
debug material stays under ignored `.artifacts/scf-active-ao/` in the measured
worktree; these retained reports and scripts are sufficient to audit the figure
and tables.
