#!/usr/bin/env bash
set -euo pipefail
issue308_dir=/home/jzzeng/codes/qc-mixed-precision/.artifacts/issue308-shell-block/v5
cd "$issue308_dir/source"
export PYTHONPATH=python:.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/tmp/vibeqc-308-gpu4pyscf-env/lib/python3.13/site-packages/cutensor/lib:${LD_LIBRARY_PATH:-}
export VIBEQC_LIBRARY="$issue308_dir/libvibeqc.so"
export VIBEQC_RESOURCE_CUDA_TEST=1 VIBEQC_DF_DERIVATIVE_CUDA_TEST=1
unset VIBEQC_DF_WEIGHTED_EXECUTION VIBEQC_DF_SHELL_SCHEDULE VIBEQC_DF_RESPONSE_ALGEBRA VIBEQC_DF_RAW_STAGING
unset VIBEQC_DF_TRACE VIBEQC_DF_HOST_TRACE VIBEQC_DF_PROGRESS_TRACE VIBEQC_DF_SHELL_COUNTERS
unset VIBEQC_DF_RESPONSE_UPLOAD_PROBE VIBEQC_DF_RESPONSE_SCATTER_PROBE VIBEQC_DF_SERIAL_RESPONSE_DOT
printf '%s\n' "$SLURM_JOB_ID" > "$issue308_dir/slurm-job.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv > "$issue308_dir/gpu-before.csv"
/tmp/vibeqc-pr-review-env/bin/python -m pytest -q tests/python/test_df_shell_derivatives_cuda.py tests/python/test_df_derivatives_cuda.py tests/python/test_df_property_budget_cuda.py > "$issue308_dir/gpu-tests.log" 2>&1
/tmp/vibeqc-pr-review-env/bin/python "$issue308_dir/qualify-defaults.py" "$issue308_dir" > "$issue308_dir/default-traced.log" 2>&1
# Restore uninstrumented fixed-density timing for the explicit oracle check.
/tmp/vibeqc-pr-review-env/bin/python -m benchmarks.issue308_response_timeline --aos 384 --batch 1 --repeats 3 --library "$VIBEQC_LIBRARY" --candidate generic --candidate combined-compact --output "$issue308_dir/clean-oracle-384ao-b1" > "$issue308_dir/clean-oracle-384ao-b1.log" 2>&1
# Five interleaved samples per engine; input/rank checks run first and exit.
/tmp/vibeqc-308-gpu4pyscf-env/bin/python "$issue308_dir/run-matrix.py" "$issue308_dir/matched" > "$issue308_dir/matched.log" 2>&1
# Direct regression evidence remains distinct from DF gates and is retained even
# if the historical absolute direct-performance acceptance is still unmet.
set +e
/tmp/vibeqc-308-gpu4pyscf-env/bin/python benchmarks/real_molecule_gate.py --density-fitting none --repeats 5 --output-directory "$issue308_dir/direct-gate" > "$issue308_dir/direct-gate.log" 2>&1
issue308_direct_status=$?
set -e
printf '%s\n' "$issue308_direct_status" > "$issue308_dir/direct-status.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv > "$issue308_dir/gpu-after.csv"
