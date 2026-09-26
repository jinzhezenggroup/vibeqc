#!/usr/bin/env bash
set -euo pipefail
issue308_dir=/home/jzzeng/codes/qc-mixed-precision/.artifacts/issue308-shell-block/v3
cd "$issue308_dir/source"
export PYTHONPATH=python:.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}
export VIBEQC_LIBRARY="$issue308_dir/libvibeqc.so"
unset VIBEQC_DF_TRACE VIBEQC_DF_HOST_TRACE VIBEQC_DF_PROGRESS_TRACE
unset VIBEQC_DF_SHELL_COUNTERS VIBEQC_DF_RESPONSE_UPLOAD_PROBE VIBEQC_DF_RESPONSE_SCATTER_PROBE
printf '%s\n' "$SLURM_JOB_ID" > "$issue308_dir/clean-slurm-job.txt"
nvidia-smi --query-gpu=name,uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv > "$issue308_dir/clean-gpu-before.csv"
for issue308_aos in 192 384 19; do
  for issue308_batch in 1 4; do
    /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.issue308_response_timeline \
      --aos "$issue308_aos" --batch "$issue308_batch" --repeats 3 --library "$VIBEQC_LIBRARY" \
      --candidate generic --candidate combined-compact --candidate combined-warp \
      --output "$issue308_dir/clean-${issue308_aos}ao-b${issue308_batch}" \
      > "$issue308_dir/clean-${issue308_aos}ao-b${issue308_batch}.log" 2>&1
  done
done
nvidia-smi --query-gpu=name,uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv > "$issue308_dir/clean-gpu-after.csv"
