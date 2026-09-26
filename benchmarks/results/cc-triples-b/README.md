# RCCSD(T) bounded CUDA tile qualification

Real-device evidence for issue #150 slice B and PR #448, executed through
Slurm on the `main` partition with `--gres=gpu:5090:1` on 2026-09-19.

## Source identity

Executed source head: `fd91b3a6b6d22ada3f4a6e34ac316a2a0226663c`.

Both JSON manifests record this head, a clean tracked worktree, the TensorIR
source identity, and SHA-256 hashes of the triples reference, tile lowering,
CUDA executor, and validator. The evidence commit is a child of the executed
source head and adds only this directory. The recorded source hashes remain
unchanged in that child.

## Device and toolchain

- NVIDIA GeForce RTX 5090, `sm_120`, 170 SMs
- Device UUID: `GPU-8e9c9e1a-e183-258c-0b3a-03a5ddebb2f8`
- NVIDIA driver: 580.95.05 (CUDA driver API version 13000)
- NVCC: CUDA 12.9, V12.9.86
- CUDA runtime: 12.9; cuBLAS: 12.9.1
- Python: 3.13.9; NumPy: 2.5.1

## Qualification scope

`manifest.json` retains H2 (near-zero triples) and H2O (nonzero triples),
including prefix-bounded partial-tile execution with full virtual summation
axes. Each configuration executes twice, using independently created resident
owners; per-tile scalars and total energy are compared bitwise. Compilation
uses the verified disk cache, while every energy is recomputed on the GPU.

| Molecule | virtual chunk | tiles | budget | total error (Eh) | max tile error (Eh) | bitwise repeat | accounted peak bytes |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| H2 | 1 | 1 | 256 MiB | 0.00e+00 | 0.00e+00 | pass | 105033288 |
| H2 | 1 | 1 | 512 MiB | 0.00e+00 | 0.00e+00 | pass | 105033288 |
| H2O | 2 | 1 | 256 MiB | 2.71e-20 | 2.71e-20 | pass | 105384496 |
| H2O | 2 | 1 | 512 MiB | 2.71e-20 | 2.71e-20 | pass | 105384496 |
| H2O | 1 | 2 | 256 MiB | 1.36e-20 | 5.42e-20 | pass | 105291312 |
| H2O | 1 | 2 | 512 MiB | 1.36e-20 | 5.42e-20 | pass | 105291312 |

Every requested run passes the total absolute-error gate <=1e-9 Eh, per-tile
gate <=1e-10 Eh, two-run bitwise equality, and plan peak <= requested budget.
The peak column is the CUDA planner's accounted device allocation, including
its library workspace allowance; it is not a measurement of process-wide GPU
memory usage. Timings and artifact keys are retained per run, with no speedup
claim. CPU/TensorIR regression tests also cover all five pinned endpoints.

`infeasible-budget.json` is a deliberately rejected H2 run at 1 MiB: the
planner requires 105033288 bytes and rejects the plan before compilation or
device allocation. The command exits **1**, preserving the failure diagnostic;
this file is rejection evidence, not a successful CUDA qualification. The
validator must fail if any requested budget is infeasible, even if another
requested budget passes.

## Reproduction

From the source checkout, set `VIBEQC_NVCC` to a CUDA 12.9 NVCC executable and
`VIBEQC_CUDA_LIBDIR` to its library directory. The executed installation was
`/home/jzzeng/cuda129-root/usr/local/cuda-12.9`.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash -lc '
    export PYTHONPATH=.:python
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    export LD_LIBRARY_PATH="$VIBEQC_CUDA_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    python tools/validate_cc_triples_tiles.py \
      --output benchmarks/results/cc-triples-b \
      --cache /tmp/vibeqc-pr448-cuda-cache \
      --nvcc "$VIBEQC_NVCC" --architecture sm_120 \
      --budget 256,512 --molecules h2,h2o
  '
```

The job preserves Slurm's assigned device visibility. To reproduce the rejected
case, use `--budget 1 --molecules h2` and a separate output directory, and expect
exit status 1.

## CPU and structural checks

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest tests/python/test_cc_triples.py \
  tests/python/test_cc_triples_tiles.py \
  tests/python/test_validate_cc_triples_tiles.py -q
```

Result: **186 passed**. Pre-commit hooks, including evidence retention,
compiler/CUDA/SCF ownership checks, Ruff, and formatting, also pass.
