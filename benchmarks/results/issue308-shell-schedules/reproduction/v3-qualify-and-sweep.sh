#!/usr/bin/env bash
set -euo pipefail
issue308_dir=/home/jzzeng/codes/qc-mixed-precision/.artifacts/issue308-shell-block/v3
cd "$issue308_dir/source"
export PYTHONPATH=python:.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}
export VIBEQC_LIBRARY="$issue308_dir/libvibeqc.so"
export VIBEQC_RESOURCE_CUDA_TEST=1 VIBEQC_DF_DERIVATIVE_CUDA_TEST=1
export VIBEQC_DF_WEIGHTED_EXECUTION=shell VIBEQC_DF_SHELL_SCHEDULE=packed
export VIBEQC_DF_RESPONSE_ALGEBRA=blas VIBEQC_DF_RAW_STAGING=pinned-panels
unset VIBEQC_DF_TRACE VIBEQC_DF_HOST_TRACE VIBEQC_DF_PROGRESS_TRACE
unset VIBEQC_DF_RESPONSE_UPLOAD_PROBE VIBEQC_DF_RESPONSE_SCATTER_PROBE
printf '%s\n' "$SLURM_JOB_ID" > "$issue308_dir/qualification-slurm-job.txt"
nvidia-smi --query-gpu=name,uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv > "$issue308_dir/gpu-before.csv"
/group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  --log-file "$issue308_dir/memcheck-%p.log" \
  /tmp/vibeqc-pr-review-env/bin/python -m pytest -q tests/python/test_df_shell_derivatives_cuda.py \
  tests/python/test_df_property_budget_cuda.py > "$issue308_dir/memcheck-tests.log" 2>&1
VIBEQC_DF_SHELL_COUNTERS=1 /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.issue308_response_timeline \
  --aos 384 --batch 1 --repeats 1 --component-trace --library "$VIBEQC_LIBRARY" \
  --candidate generic --candidate shell-warp --candidate shell-packed --candidate shell-compact \
  --candidate blas --candidate pinned --candidate combined-warp --candidate combined-packed \
  --candidate combined-compact --output "$issue308_dir/384-candidates-traced" \
  > "$issue308_dir/384-candidates-traced.log" 2>&1
