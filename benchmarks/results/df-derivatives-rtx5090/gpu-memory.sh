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
archive143=${OUTPUT_DIR:-${TMPDIR:-/tmp}/vibeqc-df-${SLURM_JOB_ID}-memory}
mkdir -p -- "$(dirname -- "$archive143")"
mkdir -- "$archive143"
printf 'OUTPUT_DIR=%s\n' "$archive143"
"$PYTHON" benchmarks/df_run_provenance.py "$archive143"

for budget143 in 0 1048576 4194304; do
 for mode143 in reference generated; do
  "$PYTHON" benchmarks/df_endpoint_memory.py --case sp8 --batch 3 --selection "$mode143" --df-budget "$budget143" --output "$archive143/memory-sp8-b3-$mode143-$budget143.json"
 done
done
for mode143 in reference generated; do
 "$PYTHON" benchmarks/df_endpoint_memory.py --case sdf18-direct --batch 1 --selection "$mode143" --df-budget 4194304 --output "$archive143/memory-sdf18-$mode143.json"
done
