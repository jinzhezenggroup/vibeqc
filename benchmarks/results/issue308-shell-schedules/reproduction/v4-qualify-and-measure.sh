#!/usr/bin/env bash
set -euo pipefail
issue308_dir=/home/jzzeng/codes/qc-mixed-precision/.artifacts/issue308-shell-block/v4
cd "$issue308_dir/source"
export PYTHONPATH=python:.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}
export VIBEQC_LIBRARY="$issue308_dir/libvibeqc.so"
export VIBEQC_RESOURCE_CUDA_TEST=1 VIBEQC_DF_DERIVATIVE_CUDA_TEST=1
export VIBEQC_DF_WEIGHTED_EXECUTION=shell VIBEQC_DF_SHELL_SCHEDULE=compact
export VIBEQC_DF_RESPONSE_ALGEBRA=blas VIBEQC_DF_RAW_STAGING=pinned-panels
unset VIBEQC_DF_TRACE VIBEQC_DF_HOST_TRACE VIBEQC_DF_PROGRESS_TRACE VIBEQC_DF_SHELL_COUNTERS
unset VIBEQC_DF_RESPONSE_UPLOAD_PROBE VIBEQC_DF_RESPONSE_SCATTER_PROBE
printf '%s\n' "$SLURM_JOB_ID" > "$issue308_dir/slurm-job.txt"
nvidia-smi --query-gpu=name,uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv > "$issue308_dir/gpu-before.csv"
/tmp/vibeqc-pr-review-env/bin/python -m pytest -q tests/python/test_df_shell_derivatives_cuda.py \
  tests/python/test_df_derivatives_cuda.py tests/python/test_df_property_budget_cuda.py \
  > "$issue308_dir/gpu-tests.log" 2>&1
for issue308_schedule in warp packed; do
  VIBEQC_DF_SHELL_SCHEDULE="$issue308_schedule" /tmp/vibeqc-pr-review-env/bin/python -m pytest -q \
    tests/python/test_df_derivatives_cuda.py::test_df_generated_sdf_bucket_preserves_all_geometry_phases \
    > "$issue308_dir/gpu-f-${issue308_schedule}.log" 2>&1
done
/group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  --log-file "$issue308_dir/memcheck-%p.log" \
  /tmp/vibeqc-pr-review-env/bin/python -m pytest -q tests/python/test_df_shell_derivatives_cuda.py \
  tests/python/test_df_property_budget_cuda.py \
  tests/python/test_df_derivatives_cuda.py::test_df_generated_sdf_bucket_preserves_all_geometry_phases \
  > "$issue308_dir/memcheck-tests.log" 2>&1
/group/software/cuda-12.9.1/bin/compute-sanitizer --tool racecheck --error-exitcode 99 \
  --log-file "$issue308_dir/racecheck-%p.log" \
  /tmp/vibeqc-pr-review-env/bin/python -m pytest -q tests/python/test_df_shell_derivatives_cuda.py \
  tests/python/test_df_derivatives_cuda.py::test_df_generated_sdf_bucket_preserves_all_geometry_phases \
  -k '(shell_execution and spherical and rhf and scalar) or sdf_bucket' \
  > "$issue308_dir/racecheck-tests.log" 2>&1
VIBEQC_DF_SHELL_COUNTERS=1 /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.issue308_response_timeline \
  --aos 384 --batch 1 --repeats 1 --component-trace --library "$VIBEQC_LIBRARY" \
  --candidate generic --candidate combined-compact --candidate combined-warp \
  --output "$issue308_dir/384-traced" > "$issue308_dir/384-traced.log" 2>&1
/group/software/cuda-12.9.1/bin/nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --cuda-event-trace=false --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop --output="$issue308_dir/nsys-384-compact" \
  /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.issue308_response_timeline \
  --aos 384 --batch 1 --repeats 1 --library "$VIBEQC_LIBRARY" --component-trace --nsys \
  --candidate combined-compact --output "$issue308_dir/384-nsys" \
  > "$issue308_dir/nsys-384-compact.log" 2>&1
/group/software/cuda-12.9.1/bin/nsys export --type=sqlite \
  --output="$issue308_dir/nsys-384-compact.sqlite" "$issue308_dir/nsys-384-compact.nsys-rep" \
  > "$issue308_dir/nsys-export.log" 2>&1
/tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_response_timeline \
  "$issue308_dir/nsys-384-compact.sqlite" --output "$issue308_dir/384-nsys-summary.json"
# Complete clean endpoints follow qualification. No compiler, instrumentation,
# diagnostic counters, sanitizer or process sampler runs during these calls.
for issue308_aos in 192 384 19; do
  for issue308_batch in 1 4; do
    /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.issue308_response_timeline \
      --aos "$issue308_aos" --batch "$issue308_batch" --repeats 3 --library "$VIBEQC_LIBRARY" \
      --candidate generic --candidate combined-compact --candidate combined-warp \
      --output "$issue308_dir/clean-${issue308_aos}ao-b${issue308_batch}" \
      > "$issue308_dir/clean-${issue308_aos}ao-b${issue308_batch}.log" 2>&1
  done
done
nvidia-smi --query-gpu=name,uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv > "$issue308_dir/gpu-after.csv"
