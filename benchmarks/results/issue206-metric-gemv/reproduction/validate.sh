#!/usr/bin/env bash
set -euo pipefail
cd /home/jzzeng/codes/vibeqc-issue-206-response-metric-gemv
[[ -n "$SLURM_JOB_ID" && -n "$CUDA_VISIBLE_DEVICES" ]]
sha256sum --check .artifacts/issue206-metric-gemv/build-success.sha256
mkdir -p .artifacts/issue206-metric-gemv/frozen-attempt1
cp -p build/cuda/libvibeqc.so build/cuda/vibeqc_density_fitting_tests build/cuda/vibeqc_df_capture_recovery_tests .artifacts/issue206-metric-gemv/frozen-attempt1/
ln -s libvibeqc.so .artifacts/issue206-metric-gemv/frozen-attempt1/libvibeqc.so.0
git diff --binary > .artifacts/issue206-metric-gemv/frozen-attempt1/source.patch
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VIBEQC_LIBRARY=$PWD/.artifacts/issue206-metric-gemv/frozen-attempt1/libvibeqc.so
export LD_LIBRARY_PATH=$PWD/.artifacts/issue206-metric-gemv/frozen-attempt1${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export VIBEQC_RESOURCE_CUDA_TEST=1 VIBEQC_DF_VALUE_MAPPING=primitive
runner=/tmp/vibeqc-pr-review-env/bin/python
printf 'Slurm job: %s\n' "$SLURM_JOB_ID"
sha256sum "$VIBEQC_LIBRARY"
"$runner" .artifacts/issue206-metric-gemv/check-identity.py
"$runner" -m pytest -q -rs --basetemp=.artifacts/issue206-metric-gemv/pytest-data tests/python/test_df_response_weights_cuda.py tests/python/test_df_force_state_export_cuda.py tests/python/test_df_device_diis_cuda.py tests/python/test_df_final_state_cuda.py tests/python/test_df_retry_eigen_cuda.py tests/python/test_hf_resources_cuda.py tests/python/test_df_property_budget_cuda.py tests/python/test_df_occupied_cuda.py
.artifacts/issue206-metric-gemv/frozen-attempt1/vibeqc_density_fitting_tests
VIBEQC_ONE_ELECTRON_DERIVATIVES=generated .artifacts/issue206-metric-gemv/frozen-attempt1/vibeqc_density_fitting_tests
VIBEQC_DF_SERIAL_RESPONSE_DOT=1 .artifacts/issue206-metric-gemv/frozen-attempt1/vibeqc_density_fitting_tests
.artifacts/issue206-metric-gemv/frozen-attempt1/vibeqc_df_capture_recovery_tests
/group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck --target-processes all --error-exitcode 86 --log-file "$PWD/.artifacts/issue206-metric-gemv/sanitizer-%p.log" "$runner" -m pytest -q -rs tests/python/test_df_device_diis_cuda.py -k occupied-4-spherical-uhf
"$runner" .artifacts/issue206-metric-gemv/profile-force.py
