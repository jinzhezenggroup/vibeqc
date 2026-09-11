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
"$PYTHON" -m pytest tests/python/test_one_electron_derivatives_cuda.py -q -x -k screened_target --junitxml="$archive141/gpu-screening.xml"
for mode141 in scalar cooperative generated_thread generated_shell_warp; do
 "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output "/tmp/issue141-profile-$mode141" "$PYTHON" benchmarks/profile_one_electron_force.py --case sp8 --batch 3 --mode "$mode141" --repeats 5 --output "$archive141/profile-$mode141.json"
 "$NSYS" stats --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_api_sum --format csv --force-export=true "/tmp/issue141-profile-$mode141.nsys-rep" > "$archive141/profile-$mode141.csv"
done
