# RCCSD #149 C native/public qualification

This directory records the endpoint qualification for issue #149 slice C. It is a correctness, lifecycle, bounded-resource, and transfer-accounting record; it is not a performance-leadership claim.

## Qualified source

- Source identity: `fc08afaf8b1e9008a57ee11cd055abca7fe9bd5e2be5be45e835ecbd103b1bd9`
- CUDA build: Release, CUDA 12.9, sm_120, production AOT shell bundles enabled
- GPU execution: RTX 5090 allocated through Slurm
- Agent: ChatGPT
- Model: GPT-5.6 Sol

Both `cpu.json` and `cuda.json` embed the same source identity, issue number, agent/model provenance, raw endpoint timings, equation hashes, convergence diagnostics, capacity, and transfer counters.

## Coverage

The matrix exercises two distinct runtime orbital shapes and three explicit correlation-memory budgets:

| case | `(nocc,nvir)` | 64 MiB CPU | 128 MiB CPU | 256 MiB CPU | 64 MiB CUDA | 128 MiB CUDA | 256 MiB CUDA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H2 / STO-3G | `(1,1)` | success | success | success | preflight reject | success | success |
| H2O / STO-3G | `(5,2)` | success | success | success | preflight reject | success | success |

The 64 MiB CUDA requests fail before the CC resident allocation because the native MO block exceeds the declared budget. This is expected bounded-resource behavior, not a fallback. At 128 and 256 MiB, H2 uses 113,247,992 B of reported numeric capacity and H2O uses 113,300,712 B.

For feasible CUDA cases, H2 agrees with the pinned independent total energy within `1.252e-12 Eh`; H2O agrees within `2.132e-13 Eh`. The CPU endpoint errors are `1.252e-12 Eh` and `2.416e-13 Eh`, respectively.

The feasible CUDA records also expose one-time setup and control movement. H2 reports 112 B setup H2D, 280 B scalar D2H, 16 B final-amplitude D2H, and 30 synchronizations over 6 CC iterations. H2O reports 11,920 B setup H2D, 752 B scalar D2H, 880 B final-amplitude D2H, and 82 synchronizations over 17 CC iterations. There is no host-driven full residual/amplitude traffic between iterations.

## Public acceptance

With the same production CUDA library, `tests/python/test_rccsd_public.py` passes all 13 CPU/CUDA public/lifecycle cases on the RTX 5090. The CPU-only run passes 10 tests with the three CUDA cases skipped. The existing MP2 public regression suite passes 27 tests with 10 environment-gated skips.

The acceptance coverage includes method-registry/public API discovery, honest energy-only capabilities, explicit unsupported force/reference/precision/DF/frozen-core behavior, zero-DIIS Jacobi, nonconvergence/denominator diagnostics, homogeneous prepared-batch repeat and partial-failure isolation, ragged-batch rejection, C-ABI owner lifetime, repeated execution, and two-context isolation.

## Reproduction

CPU qualification:

```bash
export VIBEQC_LIBRARY="$PWD/build-149-cpu/libvibeqc.so"
export PYTHONPATH="$PWD/python" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python tools/benchmark_rccsd_native.py --device cpu \
  --budget 67108864 --budget 134217728 --budget 268435456 \
  --output benchmarks/results/rccsd-149-c/cpu.json
```

CUDA qualification on the project Slurm node:

```bash
/feishu/bin/srun -p main --gres=gpu:5090:1 --cpus-per-task=4 --mem=8G bash -lc '
  cd /home/jzzeng/vibeqc-149-slicec
  export VIBEQC_LIBRARY=$PWD/build-149-cuda/libvibeqc.so
  export PYTHONPATH=$PWD/python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  VIBEQC_RCCSD_CUDA_TEST=1 python -m pytest -q tests/python/test_rccsd_public.py
  python tools/benchmark_rccsd_native.py --device cuda \
    --budget 67108864 --budget 134217728 --budget 268435456 \
    --output benchmarks/results/rccsd-149-c/cuda.json
'
```
