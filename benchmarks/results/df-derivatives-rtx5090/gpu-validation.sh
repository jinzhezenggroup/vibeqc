#!/usr/bin/env bash
set -euo pipefail
repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd -- "$repository_root"
: "${SLURM_JOB_ID:?run through a finite Slurm allocation}"
: "${CUDA_VISIBLE_DEVICES:?preserve scheduler-assigned visibility}"
printf 'SLURM_JOB_ID=%s CUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VIBEQC_LIBRARY="${VIBEQC_LIBRARY:-$PWD/build-cuda/libvibeqc.so}"
if [[ -n "${CUDA_HOME:-}" ]]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
PYTHON=${PYTHON:-python}
# Fail if a run directory already exists; archived evidence is never overwritten.
archive143=${OUTPUT_DIR:-${TMPDIR:-/tmp}/vibeqc-df-${SLURM_JOB_ID}-validation}
mkdir -p -- "$(dirname -- "$archive143")"
mkdir -- "$archive143"
printf 'OUTPUT_DIR=%s\n' "$archive143"
"$PYTHON" benchmarks/df_run_provenance.py "$archive143"

export VIBEQC_ONE_ELECTRON_DERIVATIVES=generated VIBEQC_DF_DERIVATIVES=generated
ctest --test-dir build-cuda --output-on-failure
VIBEQC_DF_DERIVATIVE_CUDA_TEST=1 "$PYTHON" -m pytest tests/python/test_df_derivatives_cuda.py -x -q --junitxml="$archive143/gpu-derivatives.xml"
VIBEQC_ONE_ELECTRON_DERIVATIVE_CUDA_TEST=1 "$PYTHON" -m pytest tests/python/test_one_electron_derivatives_cuda.py -x -q --junitxml="$archive143/gpu-one-electron.xml"
VIBEQC_CHECKPOINT_DEVICE=cuda "$PYTHON" -m pytest tests/python/test_checkpoint.py -x -q --junitxml="$archive143/gpu-checkpoint.xml"
VIBEQC_RESOURCE_CUDA_TEST=1 "$PYTHON" -m pytest tests/python/test_hf_resources_cuda.py -x -q --junitxml="$archive143/gpu-resources.xml"
VIBEQC_PROJECTION_CUDA_TEST=1 "$PYTHON" -m pytest tests/python/test_basis_projection_cuda.py -x -q --junitxml="$archive143/gpu-projection.xml"
