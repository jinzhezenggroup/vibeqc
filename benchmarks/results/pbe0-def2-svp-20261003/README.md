# PBE0/def2-SVP: complete strict energy and force endpoints

This historical convergence-repair baseline is retained unchanged. The newer
[grid-reuse campaign](../pbe0-grid-reuse-20261003/README.md) measures the subsequent
performance changes against freshly measured independent references.

![All-five warm medians and ranges](pbe0.svg)

All six sizes now complete cold, changed-geometry and ten warm endpoints,
without relaxing the scientific gates. These are public default FP64 energy
**and analytic-force** timings on n1 RTX 5090 GPUs. Every warm observation
remains in the median and min/max bars; incomplete attempts are not plotted.

## Complete warm medians, seconds

| Atoms | Spherical AOs | GenerativeQC | GPU4PySCF | Native / reference |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 24 | 0.365642 | 1.251881 | 0.29× |
| 6 | 48 | 0.807596 | 2.052941 | 0.39× |
| 12 | 96 | 2.043103 | 1.390916 | 1.47× |
| 24 | 192 | 6.641760 | 2.136671 | 3.11× |
| 48 | 384 | 24.925574 | 5.949080 | 4.19× |
| 96 | 768 | 106.987277 | 14.048509 | 7.62× |

**The large-system performance gap is reduced, not eliminated.** Native is
faster at 3/6 atoms but remains slower at 12–96 atoms. All 60 native warm calls
take one SCF iteration, with no cold fallback. Reference original warm calls
take 1/3/1/1/3/3 iterations at the six sizes. Moved reference replays also vary,
including 1–8 iterations at 96 atoms. Timings are not divided by iterations.

## Same-host pre-change controls

| Atoms | Control warm | Candidate warm | Endpoint speedup |
| ---: | ---: | ---: | ---: |
| 3 | 0.560805 | 0.365642 | 1.53× |
| 12 | 3.451568 | 2.043103 | 1.69× |
| 24 | 12.151722 | 6.641760 | 1.83× |
| 48 | 58.764439 | 24.925574 | 2.36× |

Each control completes all 12 endpoints against the identical reference file.
These use the retained pre-change native binary/runtime, not the much older
October 1 measurements. They include the benchmark-only reference repair but
no snapshot cache, cooperative default or fused full-range derivative source.
No completed 96-atom control speedup is claimed. See `control-water<atoms>.json.gz`.

## Cold and changed geometry, seconds

| Atoms | Native cold | Reference cold | Native moved | Reference moved | Native moved warm | Reference moved warm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 57.821609 | 23.150607 | 1.966426 | 10.021558 | 0.365528 | 1.260402 |
| 6 | 52.797222 | 18.342832 | 4.184027 | 11.582931 | 0.816024 | 1.970658 |
| 12 | 64.082789 | 18.743998 | 10.621878 | 18.631150 | 2.055503 | 2.238223 |
| 24 | 91.737691 | 32.125573 | 28.922033 | 29.069852 | 6.640092 | 2.142426 |
| 48 | 193.881896 | 57.465039 | 82.821522 | 49.792381 | 25.132565 | 4.021435 |
| 96 | 547.364966 | 94.419539 | 289.353391 | 87.184042 | 106.702673 | 10.512101 |

Cold includes preparation and synchronized execution, including required CUDA
compilation/cache setup. Imports, binary hashing and setup device probes are
outside the timer. Caches are not purged between sizes; cold differences are
not isolated kernel speedups. Moved includes geometry invalidation/preparation
and reconvergence. Both engines return host forces before stopping the clock:
native uses `execute(properties=("energy", "forces"))`; GPU4PySCF includes
analytic grid response, force download and final synchronization.

## Convergence repairs and scientific gates

Two different issues required separate repairs:

1. GPU4PySCF 1.8.1 consumes incremental `dm_last`/`vhf_last` even when
   `direct_scf=False`. The reference now rebuilds from the complete current
   density on every `get_veff` call; the work is timed and counted. All six
   reference protocols converge. GPU4PySCF still owns its integrals, XC,
   SCF and analytic derivatives; no reference density enters native execution.
2. Naive scalar CUDA energy traces lose sub-ULP signed AO contributions above
   the strict stopping threshold. Before compensation, one 96-atom attempt
   times out at 3600 s during changed geometry; another completes cold in
   61 iterations but times out in its first warm replay. Neumaier-compensated
   traces preserve these contributions without changing the operator or
   stopping policy. Final 96-atom cold/moved calls take 26/12 iterations,
   followed by ten one-step replays. Earlier timeouts remain separate evidence.

All **72 native endpoints** pass independent absolute gates of `1e-8 Eh` and
`1e-7 Eh/Bohr`; observed maxima are **`1.0141e-10 Eh`** and
**`3.1044e-11 Eh/Bohr`**. All 72 reference endpoints also pass their original/
moved consistency gates. Every repeat is checked, not just the median.

- Neutral RKS water32mer prefixes; complete spherical def2-SVP with the
  offline [H/O snapshot](def2-svp-ho.json), canonical basis identity
  `fad2c7433fe5fac19525d4c80fbd995a7ebed7816b7ea4bb88547e4f6a58a3d8`.
- Identical unpruned moving grid: 48 rational-Legendre radial × 16 Legendre
  polar × 32 trapezoidal azimuth points per atom, radius 1 Bohr, three original
  Becke iterations, no radii adjustment or small-density grid deletion.
- Native SCF energy/density tolerances `1e-12 Eh`/`1e-10`, screening `1e-12`,
  maximum 100 iterations. Reference energy/gradient tolerances `1e-12 Eh`/
  `1e-10`, direct screening `1e-14`, maximum 100 iterations. Equal thresholds
  do not imply identical stopping algorithms.
- Each engine owns its cold density and five frozen post-cold replays. Atom
  two moves +0.001 Bohr in z, reconverges and supplies five frozen moved replays.
- No density fitting, mixed precision, VV10 or opt-in native schedule. Public
  SCF and force tiles remain 256 points; experimental larger tiles are excluded.

## Work, resources and remaining bottlenecks

The exact snapshot cache retains one immutable grid under a charged 128 MiB
host cap. It checks every point/weight/owner and the live native lease; changed
geometry replaces the entry and over-budget grids stay uncached. It does not
cache densities, forces or solver state.

The full-range derivative provider shares J/K shell traversal and downloads
both channels together. Generic high-order derivatives are evaluated once;
specialized low-order recurrences still execute separately. Existing cooperative
Becke execution is automatically requested only on sm_120 for 12–128 atoms,
subject to native resource checks. Other targets, small systems and explicit
generic selection retain the fallback.

The large geometry workload remains. At 48/96 atoms the summaries record
1,179,648/2,359,296 grid points, 4,608/9,216 geometry batches and
2,661,287,016/21,516,784,080 partition grid-pair visits. At 96 atoms the chunk
planner decreases from 16,384 to 10,752 points under the unchanged budget.
Raw work summaries retain these counts and actual resource bounds.

Provider-internal two-to-one traversal/download reductions are source-audited,
not inferred from coarser snapshot launch counters. Logical quartet capacity
is not a post-screen integral count. Missing counters/timings remain empty or
`null`; zero optional device timers are not measured speedups. Full endpoint
wall time remains the performance authority.

## Provenance and qualification

[provenance.json](provenance.json) binds source, binary and scheduler records.
Production source is `017284e49`, incorporating master `06459d469`; subsequently
fetched `db44f7939` only changes unrelated CC arena code and is not relabeled
as the measured binary. Remote sources use `git archive` plus the recorded
patch. Raw Git metadata remains `null`; all inventoried production hashes
match the recorded commit.

- Native library SHA-256:
  `a69b6a4aef8703ca1dcd35c3d61192658c4410c02f6dd321c88b37423a67c858`.
  Native source identity:
  `d05113beea81c5f4ae67a81a2c733cae8b234ef24e90a043ce5eca8af376a641`.
- Release CUDA 12.9.1 / sm_120, generated shell AOT enabled, explicit C++/CUDA
  `ccache` launchers, scalar CPU linear-algebra fallback; eight OpenMP/BLAS/MKL
  threads. GPU4PySCF 1.8.1, PySCF 2.14.0, CuPy 13.6.0, NumPy 2.4.6,
  cuTENSOR 2.2.0 and CUDA LibXC 7.0.0.
- All accepted timings use n1 RTX 5090s in finite `main` Slurm allocations
  with `--gres=gpu:5090:1`, preserving assigned visibility. Final native jobs
  are 5362 (96 atoms) and 5363 (3–48); reference jobs are 5328/5329/5330 and
  control job is 5333. Other allocated GPUs are not timing samples.
- The corrected library passes 19 native/snapshot, 14 independent cooperative
  force and six hybrid analytic/reconverged finite-difference tests. Exact-sum
  host/NVCC regressions and physical diagnostics pass focused memcheck:
  27 passed, one skipped, zero errors. One pre-existing CUDA history assertion
  fails identically on the unchanged integration binary: it expects `None`
  rather than the retained first-warm energy baseline. It is not counted as
  passed. Unchanged large-geometry kernels pass 96/128-atom memcheck, initcheck
  and synccheck on the separately recorded integration build.
- Review follow-up in Slurm job 5380 passes six host and 21 NVCC trace probes
  under memcheck with zero errors. These cover explicitly rounded nonbinary
  products, unequal spins, full/range cancellation, null exchange pointers,
  failure bits and nonfinite rejection. The production kernel is unchanged.
- `water<atoms>.json.gz` (lossless gzip)
  retains every scalar observation, gates, source hashes,
  work/resources and both independent force oracles. Corresponding raw
  `*-{native,reference}.json.gz` files retain **all** forces; decompressed hashes
  must match `raw_sha256`. Controls and the two failed 96-atom attempts stay
  separate. Historical build/profile logs remain in ignored
  `.artifacts/pbe0-large-20261002/`, not external release assets.

Eleven formerly plain complete/control/failed reports are now losslessly
compressed too. [report-storage.json](report-storage.json) records both byte
counts and hashes, plus the immutable original Git revision. Decompression
recovers every original byte, not merely an equivalent JSON object. No sample,
failed outcome, numerical gate or measured source identity changes. The existing
96-atom and raw endpoint gzip members are unchanged. Use `gzip -dc <file.json.gz>`
to inspect a report; `tests/python/test_pbe0_report_storage.py` checks every new
member. This storage-only change saves 614,539 payload bytes before the manifest.

## Reproduce

With the recorded library, GPU Python environment and CUDA dependencies
configured, run each size in a finite allocation:

```bash
export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
export PBE0_BENCHMARK_PYTHON="$PWD/.artifacts/gpu-env/bin/python"
export PBE0_BASIS_FILE="$PWD/benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json"
export PBE0_BENCHMARK_OUTPUT="$PWD/.artifacts/pbe0-reproduction"
export PBE0_POINT_TIMEOUT=3600
for atoms in 3 6 12 24 48 96; do
  PBE0_ATOMS="$atoms" srun --partition=main --gres=gpu:5090:1 \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --time=01:20:00 \
    bash benchmarks/run_pbe0_benchmarks.sh
done
PYTHONPATH=python:. "$PBE0_BENCHMARK_PYTHON" -m tools.render_readme_pbe0 \
  --raw-directory "$PBE0_BENCHMARK_OUTPUT" \
  --basis-file "$PBE0_BASIS_FILE" --destination .artifacts/pbe0-rendered
```

Do not combine these reference-policy timings with the historical October 1
curve or promote partial diagnostics into complete endpoints.
