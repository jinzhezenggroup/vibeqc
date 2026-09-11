#!/usr/bin/env bash
set -euo pipefail
repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd -- "$repository_root"
: "${SLURM_JOB_ID:?run inside Slurm}"
: "${CUDA_VISIBLE_DEVICES:?preserve scheduler visibility}"
printf 'SLURM_JOB_ID=%s CUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
export PYTHONPATH=python:.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VIBEQC_LIBRARY="${VIBEQC_LIBRARY:-$PWD/build-cuda/libvibeqc.so}"
if [[ -n "${CUDA_HOME:-}" ]]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
export VIBEQC_ONE_ELECTRON_DERIVATIVE_CUDA_TEST=1
PYTHON=${PYTHON:-python}
NSYS=${NSYS:-nsys}
archive141=benchmarks/results/one-electron-derivatives-rtx5090
ctest --test-dir build-cuda --output-on-failure
"$PYTHON" -m pytest tests/python/test_one_electron_derivatives_cuda.py -x -q --junitxml="$archive141/gpu-derivatives.xml"
VIBEQC_CHECKPOINT_DEVICE=cuda VIBEQC_ONE_ELECTRON_DERIVATIVES=generated "$PYTHON" -m pytest tests/python/test_checkpoint.py -x -q --junitxml="$archive141/gpu-checkpoint.xml"
VIBEQC_RESOURCE_CUDA_TEST=1 VIBEQC_ONE_ELECTRON_DERIVATIVES=generated "$PYTHON" -m pytest tests/python/test_hf_resources_cuda.py -x -q --junitxml="$archive141/gpu-resources.xml"
