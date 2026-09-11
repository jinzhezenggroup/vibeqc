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
PYTHON=${PYTHON:-python}
NSYS=${NSYS:-nsys}
archive141=benchmarks/results/one-electron-derivatives-rtx5090
for mapping141 in thread shell_warp; do
 "$PYTHON" benchmarks/one_electron_values_gate.py --derivatives --case sp8 --batch 3 --mapping "$mapping141" --observe-resources --output "$archive141/sp8-b3-$mapping141.json"
done
"$PYTHON" benchmarks/one_electron_values_gate.py --derivatives --case sdf18-direct --mapping thread --output "$archive141/sdf18-thread.json"
"$PYTHON" benchmarks/one_electron_values_gate.py --derivatives --case sp8 --contraction-length 8 --mapping thread --observe-resources --output "$archive141/sp8-long8-thread.json"
"$PYTHON" benchmarks/one_electron_values_gate.py --derivatives --case sp8 --batch 3 --mapping thread --fitted --observe-resources --output "$archive141/sp8-df-b3-thread.json"
"$PYTHON" benchmarks/one_electron_values_gate.py --derivatives --case sp8 --batch 3 --mapping shell_warp --fitted --df-budget 1048576 --observe-resources --output "$archive141/sp8-df-b3-streamed.json"
