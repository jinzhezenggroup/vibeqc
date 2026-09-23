# README endpoint benchmarks (2026-09-22)

**Draft: benchmarking and publication are paused.** The GPU correctness,
performance and capability blockers are tracked in
[#1077](https://github.com/jinzhezenggroup/vibeqc/issues/1077),
[#1078](https://github.com/jinzhezenggroup/vibeqc/issues/1078),
[#1079](https://github.com/jinzhezenggroup/vibeqc/issues/1079),
[#1080](https://github.com/jinzhezenggroup/vibeqc/issues/1080) and
[#1081](https://github.com/jinzhezenggroup/vibeqc/issues/1081).
See the [diagnosis note](../../../.agents/notes/proposed/2026-09-22-gpu-benchmark-blockers.md)
for measured phases, work counts and the relationship to PR #1076.

These measurements use master `572fbd6d4cfc3dd0db03fb0d1abaac4add0b9f12`.
The benchmark drivers and plotting tool are included in this change; production
methods are unchanged. [summary.json](summary.json) binds the
[HF](hf.json), [DFT](dft.json) and [CCSD(T)](ccsd-t.json) sample records by
checksum, retaining every measured latency, convergence branch, error gate,
binary hash and raw-file checksum.

## Scope

| Series | VibeQC | Reference | Endpoint |
| --- | --- | --- | --- |
| RHF direct | CUDA | GPU4PySCF 1.8.1 | Energy + analytic forces |
| RHF DF | CUDA, cc-pVDZ-JKFIT | GPU4PySCF, identical auxiliary basis | Energy + analytic forces |
| PBE / r²SCAN direct | CUDA | GPU4PySCF, identical explicit grid | SCF energy |
| PBE0 direct | CUDA unavailable | GPU4PySCF | SCF energy, reference only |
| PBE / r²SCAN / PBE0 DF | DF-DFT unavailable | GPU4PySCF, cc-pVDZ-JKFIT | SCF energy, reference only |
| CCSD(T) | Internal CUDA composition | Pinned independent PySCF energies (validation only) | Fresh GPU RHF + host preparation + resident CUDA CCSD + bounded CUDA (T) |

The DF-DFT and CUDA hybrid omissions are explicit public API rejections on the
measured revision. Public native CUDA CCSD(T) is also unavailable. The internal
composed CUDA helper is timed under its separate, explicit execution contract.
An unavailable result is not a zero-time result or a performance comparison.

## Scientific and timing protocol

- HF and DFT use spherical def2-SVP and nested prefixes of the existing
  `water-32mer-4s4-def2-svp-spherical` topology. HF covers 3–96 atoms
  (24–768 AOs); DFT uses the same 3–96 atom sizes, with batch size one.
- DF uses the complete, committed
  [cc-pVDZ-JKFIT snapshot](../issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json),
  shared by both engines. It is an explicit auxiliary choice with a different
  approximation from direct integrals, not an assertion of DF/direct equality.
  DF reference SCF rebuilds the full Fock rather than incremental density updates.
- SCF energy/density tolerances are `1e-10`/`1e-9`; the independent reference
  orbital-gradient tolerance is `1e-8`, with at most 100 cycles. Native
  screening is `1e-12`; reference direct-SCF screening is `1e-14`.
- HF uses the existing audited comparator. Every measured repeat must converge
  and pass `|ΔE| ≤ 1e-8 Eh` and `max|ΔF| ≤ 1e-7 Eh/Bohr` against GPU4PySCF.
  The DF comparison includes the reference auxiliary-basis force response.
- DFT uses PBE's resolved production grid and r²SCAN's current default grid;
  PBE0 uses explicit `GridSpec()` because the hybrid production grid policy is
  not promoted. Both engines consume exactly the same points and weights, with
  no pruning or small-density point removal. Counts, full grid specifications
  and identities are retained. This tests the same discrete energy; it is not
  an independent grid-refinement study. Supported VibeQC results must satisfy
  `|ΔE| ≤ 1e-8 Eh` for cold, priming and every measured repeat.
  Reference grid export uses the public grid iterator with 32,768-point host
  tiles and an optional geometry/specification-keyed cache. Export occurs before
  solve timing; a separate test verifies exact equality with the small-grid
  exporter, including cache reload. Native grid execution is unchanged.
- HF/DFT warm timing reuses a prepared plan/object and a **frozen engine-local
  post-cold density**. One unmeasured priming replay precedes three interleaved
  repeats per engine. CUDA work is synchronized at endpoint boundaries. Cold
  time and preparation boundaries are recorded separately in the raw artifacts.
- Plots show a median and min/max only within a stable per-engine SCF iteration
  branch. Crosses show individual observations if a branch is unstable; those
  observations are not pooled into a headline median. Cross-engine branch counts
  can differ, so the charts report latency, not equal-work speedups.
- CCSD(T) uses the committed H₂O, NH₃ and CH₄ STO-3G geometries and exact shell
  parameters, all electrons active. Each sample starts fresh GPU RHF, constructs
  a canonical snapshot and a conventional integral provider, solves resident
  CUDA CCSD, and evaluates bounded CUDA (T) tiles. No amplitudes or transformed
  tensors are retained between samples; compiled artifacts are reused.
- The CCSD(T) interface is
  `tools.vibeqc_cc.ccsd_t_api.rccsd_t_energy(backend="cuda-resident")`.
  Its existing bridge canonicalizes on the host and uses CPU AO/MO preparation;
  these operations, transfers and owner cleanup are **inside** the timer. This is
  an internal CUDA composition, not a public/native all-GPU implementation.
  The snapshot bridge is limited to 12 AOs, so these small cases do not establish
  realistic large-system CC scaling. GPU4PySCF 1.8.1 has no `(T)` endpoint, and
  no CPU performance series is shown as its substitute.
- CC convergence uses `1e-12 Eh` energy change and `1e-10` physical residuals.
  Independent pinned PySCF 2.14.0 CCSD and (T) values check every cold/warm total
  energy at `1e-8 Eh` and the separate (T) correction at `1e-10 Eh`; no oracle
  executes inside the measured CUDA path. RHF/CC
  iterations, expanded physical residual replay, triangular virtual-triple
  counts, provider transformations/source tiles, transfers and CUDA artifact
  identities are retained. The first call in each process is recorded separately
  from the three subsequent complete-endpoint calls. The persistent JIT cache
  may already contain artifacts from validation runs; this first-call number
  does not claim a clean-cache compilation measurement.

## Hardware and build

GPU runs use one NVIDIA GeForce RTX 5090 exclusively allocated by Slurm
(`main`, `--gres=gpu:5090:1`), preserving `CUDA_VISIBLE_DEVICES`. The host is an
AMD EPYC 7K62 (48 exposed CPU cores). Each benchmark sets OpenMP, OpenBLAS and
MKL thread limits to eight. Host work used by the internal CC composition
remains part of its measured endpoint.

CUDA uses the unmodified `cuda-release-sm120` preset (Release, production AOT
manifest, no fast compile), CUDA 12.9.1, PySCF 2.14.0, GPU4PySCF 1.8.1 and
CuPy 14.2.0. CCSD(T) additionally uses the existing generated resident/tiled
CUDA owners through an explicit `sm_120` compiler adapter.
Actual library hashes, source identities, package versions and device state are
recorded with the measurements. No timing is taken from an older native binary.

## Reproduce

These commands reproduce diagnostic records, not a qualified publication.
Known large-system points can still fail or hit the per-point timeout; the
runner retains those failures and exits nonzero. Reassess #1117's independent
numerical and complete-endpoint gates before publishing any new figures.

From the repository root, use a Python environment with NumPy, PySCF 2.14.0,
GPU4PySCF CUDA 12 1.8.1 and CuPy CUDA 12 14.2.0:

```bash
cmake --preset cuda-release-sm120 \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DPython3_EXECUTABLE="$(command -v python)" -DVIBEQC_BUILD_TESTS=OFF
cmake --build --preset cuda-release-sm120 -j10
export PATH=/group/software/cuda-12.9.1/bin:$PATH
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}
export VIBEQC_LIBRARY=$PWD/build/cuda-release-sm120/libvibeqc.so
export README_BENCHMARK_OUTPUT=$PWD/.artifacts/readme-reproduction
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=04:00:00 bash benchmarks/run_readme_benchmarks.sh hf
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=02:00:00 bash benchmarks/run_readme_benchmarks.sh dft
```

Run CCSD(T) in its own finite GPU allocation using the same CUDA library:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=01:00:00 bash benchmarks/run_readme_benchmarks.sh cc
```

Generate the figures with Matplotlib from the resulting raw directory:

```bash
python tools/render_readme_benchmarks.py \
  --raw-directory .artifacts/readme-reproduction \
  --destination .artifacts/readme-reproduction-figures
```

Full logs, progress journals, endpoint result arrays and failed attempts remain
in ignored `.artifacts/readme-benchmarks-20260922/`. The compact summary binds
each accepted or unavailable point to its original bytes by SHA-256. No release,
release asset or external archive was published for this update.
