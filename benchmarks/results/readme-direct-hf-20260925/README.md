# HF direct J/K endpoints (2026-09-25)

These are the **previously completed direct HF measurements**, not a new run
on the commit containing this README update. The six source records report
clean `master` commit `fe534ebf563108aecadcd3b967c7652f6d62f79f`,
production Release `sm_120` with shell AOT enabled, and native library SHA-256
`96c84d9daebf1f427cbf715136529a4afe729cc79ffc18ba25ca83c7729a9a8f`.
The PR containing this result is based on a **later** `master`; do not treat
these figures as a measured speedup of that later source revision. Hardware:
one Slurm-allocated NVIDIA GeForce RTX 5090; CUDA 12.9.1, PySCF 2.14.0,
GPU4PySCF 1.8.1, CuPy 14.2.0. OpenMP, OpenBLAS and MKL were set to eight
threads each. No DF result is included in this PR.

| Atoms / spherical AOs | VibeQC (ms) | GPU4PySCF (ms) |
| --- | ---: | ---: |
| 3 / 24 | 84.2 | 879.4 |
| 6 / 48 | 89.8 | 934.0 |
| 12 / 96 | 129.0 | 1,081.3 |
| 24 / 192 | 263.0 | 1,478.5 |
| 48 / 384 | 758.6 | 2,128.5 |
| 96 / 768 | 3,054.1 | 3,544.3 |

Each entry is the median of three interleaved, synchronized **complete warm
SCF energy-plus-analytic-force** endpoints, with min/max error bars in the
figure. Both engines use the same nested water geometries, spherical def2-SVP,
batch size one, and their own fixed post-cold density for each warm replay.
Cold preparation and SCF/forces are measured separately and excluded from the
plotted warm latency. The comparator uses GPU4PySCF's ordinary incremental
direct Fock and VibeQC's ordinary direct J/K policy; this is an endpoint
comparison, not an assertion of identical internal work. Both reported one
converged SCF iteration on every warm repeat at all six sizes.

All six cases passed the independent GPU4PySCF comparison gates
`|ΔE| ≤ 1e-8 Eh` and `max|ΔF| ≤ 1e-7 Eh/Bohr`. Maximum observed errors were
`2.32e-11 Eh` and `1.24e-9 Eh/Bohr`. The compact [per-repeat
records](hf.json) retain cold/warm times, convergence and work counters,
accuracy gates, source and basis identities; [summary.json](summary.json)
binds them to SHA-256 hashes of the original complete raw JSON. Raw point
records and progress journals remain in ignored local
`.artifacts/readme-hf-df-aot-20260925/hf/direct-*.json` and
`.artifacts/readme-hf-df-aot-20260925/hf/direct-*.progress.jsonl`.

## Run a new qualification

To measure a **different** source revision, build its production Release
`sm_120` library with `VIBEQC_ENABLE_AOT_SHELLS=ON`, set `VIBEQC_LIBRARY` to
that library, and run only the direct HF matrix under a finite GPU allocation:

```bash
export README_BENCHMARK_PYTHON=/path/to/benchmark-env/bin/python
export VIBEQC_LIBRARY=$PWD/build/cuda-release-sm120/libvibeqc.so
export README_BENCHMARK_OUTPUT=$PWD/.artifacts/readme-direct-hf-rerun
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=01:00:00 bash benchmarks/run_readme_benchmarks.sh hf-direct
python tools/render_readme_benchmarks.py \
  --raw-directory .artifacts/readme-direct-hf-rerun \
  --destination .artifacts/readme-direct-hf-rerun-figures
```

The runner rejects missing library configuration, retains per-point logs and
progress journals, and fails if a case times out or misses its numerical gate.
It leaves Slurm-assigned device visibility unchanged. Do not combine such a
rerun with the historical results above without labeling its new source hash.
