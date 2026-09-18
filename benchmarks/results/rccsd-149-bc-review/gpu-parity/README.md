# #149 B Resident-vs-Ordinary GPU Parity Evidence

## Current reviewed source

`rtx5090-review-20260917.json` retains all eight runtime cases for reviewed
source `4c467810b04418b673223e64e801be7eedf86ece`, with three profiled/unprofiled
replays per case, exact resident artifact keys and binary hashes, compiler
source identity, and the runtime CUDA device UUID/architecture. All 24 parity
replays and stale-lease/nonfinite/missing-input/repeat checks passed on the
Slurm-allocated RTX 5090 (sm_120, CUDA runtime 12.9). Compute Sanitizer memcheck
repeated all eight cases with zero errors. The review changes only repair the
ownership inventory; measured compiler/runtime/validator bytes are unchanged.

Reproduce on this workstation through the scheduler:

```sh
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:20:00 \
  env PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python tools/validate_cc_resident.py --output /tmp/resident-results \
  --cache /tmp/resident-cache --nvcc /path/to/cuda-12.9/bin/nvcc \
  --architecture sm_120
```

## Historical H200 observation

Device: NVIDIA H200 (141GB), sm_90, CUDA 12.8, Driver 570.124.06
NVCC: Cuda compilation tools, release 12.8, V12.8.61
Python: 3.11.16, NumPy 2.2.6
Commit: f2134c7 (claude/issue-0149-b)

## Results

4 molecules × 2 budgets (normal 256MiB / minimum) = 8 runtime cases.
Every case compares the ordinary host-staged path against the resident
inlined-kernel path on identical feeds.

| Case | Budget | Parity (3 runs) | Stale Lease | Nonfinite | Missing Input | Repeat |
|------|--------|-----------------|-------------|-----------|---------------|--------|
| random (2,2) | normal | ✓ | ✓ | ✓ | ✓ | ✓ |
| random (2,2) | minimum | ✓ | ✓ | ✓ | ✓ | ✓ |
| h2 (1,1) | normal | ✓ | ✓ | ✓ | ✓ | ✓ |
| h2 (1,1) | minimum | ✓ | ✓ | ✓ | ✓ | ✓ |
| water (5,2) | normal | ✓ | ✓ | ✓ | ✓ | ✓ |
| water (5,2) | minimum | ✓ | ✓ | ✓ | ✓ | ✓ |
| lih (2,4) | normal | ✓ | ✓ | ✓ | ✓ | ✓ |
| lih (2,4) | minimum | ✓ | ✓ | ✓ | ✓ | ✓ |

All 8 cases were reported passing (24 parity replays, 32 error/repeat checks).
This historical observation does not qualify later compiler revisions.

## Full evidence

The per-case JSON records (including per-run metrics, artifact keys and
binary hashes) are retained on the qz shared filesystem at:
  /inspire/qb-ilm/project/chemicalreaction/czxs25220150/scratch/vibeqc/issue-0149-b/results/gpu/

Manifest SHA256: eae15410ef551a27bbab9dfde5b11d44b8baa3c826d20511b9102e8b2c1c02bc

## Not-run

- GPU-job / multi-GPU batching
- Non-RCCSD programs
