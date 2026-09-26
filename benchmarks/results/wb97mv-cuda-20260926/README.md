# WB97M-V CUDA force qualification (2026-09-26)

Complete **public Python CUDA energy and analytic forces** pass the independent
numerical gates below. Scaling performance is **not qualified**. This evidence
is retained for review and future work; it is not a headline README benchmark.

The implementation was measured in a dirty worktree based on
`872b52f5de002787d68a3d5c724fd1224572ce36`. The loaded Release `sm_120` native
library SHA-256 is
`551ae03cd5fc8e0db6691d413ef4e060568f26d85e3f5e7733cd097030df557b`.
Shell and stationary-force AOT were enabled. The [source file hashes](measured-source-files.json)
identify the modified scientific sources used for this retained acceptance run.
These timings predate the later quartet-driven direct-JK derivative scheduling
optimization on the PR branch, so they are a baseline for the old kernel only
and must not be attributed to the current head. The retained binary identity
must not be confused with either unmodified master or the later optimized head.

## Numerical acceptance

Four complete CUDA tests pass: H2/RKS/STO-3G, H3/UKS/STO-3G, spherical
water/RKS/def2-SVP, and geometry rebuild/failure isolation/stale-token recovery.
The force cases compare GPU4PySCF with moving-grid response, check translation,
and compare random internal directional derivatives against reconverged native
energies at three finite-difference steps. These tests use a 12 × 4 × 8 atomic
grid. Independent native SR/LR derivative tests cover s/p/d/f, both AO layouts,
both spin modes and two geometries. Eight VV10 CPU/CUDA tests, three CPU
WB97M-V regressions and 62 compiler regressions also pass. See the exact
[validation receipts and gates](validation.json).

The default-grid three-atom benchmark independently passes `|dE| <= 1e-8 Eh`
and `max|dF| <= 1e-7 Eh/Bohr` across cold, priming and all three warm pairs.
Maximum errors are `4.73e-9 Eh` and `4.76e-9 Eh/Bohr`.

## Endpoint measurements and incomplete attempts

| Atoms / AOs | VibeQC warm (s) | GPU4PySCF warm (s) | Paired attempt |
| --- | ---: | ---: | --- |
| 3 / 24 | 19.216 | 2.310 | Complete; accuracy accepted |
| 6 / 48 | — | 8.421 | 150 s point timeout during native cold execution |
| 12 / 96 | — | — | Stopped during native cold execution |
| 24 / 192 | — | — | Not run |
| 48 / 384 | — | — | Not run |
| 96 / 768 | — | — | Not run |

The 6-atom comparator is an independent reference-only measurement, so it does
not imply a passing paired accuracy gate. Reference-only 12/24/48/96-atom
attempts each timed out during cold execution at a **150 s whole-point** limit.
The remaining native matrix was stopped at the user's request after the slow
initial result. Timeouts and stopped points are not endpoint timings or bounds
on warm latency. Performance analysis and optimization are deferred.

The [three-atom paired record](water3.json) and [six-atom reference record](water6-reference.json)
retain full forces, convergence and repeats. The [summary](summary.json) retains
every attempt's status, stage, raw SHA-256 and original ignored artifact path,
plus source/binary identities, environment and derivative work counts. No
uncompleted repeat contributes a median.

The protocol follows the direct-HF README: nested water geometries, spherical
def2-SVP, batch size one, synchronized complete SCF energy plus analytic-force
endpoints, each engine's own fixed post-cold density, an unmeasured priming run,
and three interleaved warm repeats. The common grid has 48 radial × 16 polar ×
32 azimuthal points per atom, with original three-step Becke partitioning,
no radius adjustment or pruning, and analytic grid response in both engines.
Complete WB97M-V includes self-consistent VV10 and exact range-separated
exchange. Native warm repeats converged in two SCF iterations; GPU4PySCF used
one. This compares each engine's complete endpoint, including its ordinary
convergence policy. Compiler caches may already be populated by acceptance
tests; the warm latency excludes cold calculator preparation/execution.

Hardware/software: one Slurm-allocated RTX 5090, CUDA 12.9.1, PySCF 2.14.0,
GPU4PySCF 1.8.1 and CuPy 14.2.0. OpenMP/OpenBLAS/MKL each use eight threads.
GPU4PySCF uses its CuPy contraction fallback, as in the retained direct-HF
baseline. The native production force path uses no CPU/reference derivatives.
Host snapshot exports, tiling and total-density/VV10-active-set packing remain
explicit in the force work record.

## Reproduce

Follow the [public contract and build/runtime requirements](../../../docs/user/wb97mv_cuda.md).
Use the same source checkout for the Python package and Release native library.
Every real-GPU command must run under a finite Slurm allocation:

```bash
export README_BENCHMARK_PYTHON=/path/to/benchmark-env/bin/python
export VIBEQC_LIBRARY=$PWD/build/libvibeqc.so
export README_BENCHMARK_OUTPUT=$PWD/.artifacts/wb97mv-rerun
export CUDA_PATH=/path/to/cuda-12.9
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:15:00 \
  env PYTHONPATH=python:. OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  "$README_BENCHMARK_PYTHON" -m benchmarks.readme_wb97mv --atoms 3 --repeats 3 \
  --output "$README_BENCHMARK_OUTPUT/wb97mv/direct-3.json"
```

The finite full-matrix runner remains available as
`bash benchmarks/run_readme_benchmarks.sh wb97mv` inside a sufficiently long
Slurm allocation. It defaults to 900 s per point; the retained interrupted
campaign used 600 s for paired water3 and 150 s for all other attempted points.
`wb97mv-reference` measures the comparator independently. Preserve Slurm's
assigned device visibility.
