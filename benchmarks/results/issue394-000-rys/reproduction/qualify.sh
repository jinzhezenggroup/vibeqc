#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VIBEQC_RESOURCE_CUDA_TEST=1
export VIBEQC_LIBRARY="$PWD/.artifacts/issue394-000/v1/libvibeqc.so"
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64
export VIBEQC_DF_SHELL_MATH_000=rys
build/cuda-release-sm120/vibeqc_df_shell_pairs_tests > .artifacts/issue394-000/v1/native-shell.log 2>&1
/group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck --error-exitcode 99 build/cuda-release-sm120/vibeqc_df_shell_pairs_tests > .artifacts/issue394-000/v1/memcheck.log 2>&1
/tmp/vibeqc-pr-review-env/bin/python .artifacts/issue394-000/endpoint.py
export VIBEQC_DF_SHELL_MATH_000=rys
/tmp/vibeqc-pr-review-env/bin/python -m pytest -q -x tests/python/test_df_shell_derivatives_cuda.py tests/python/test_df_response_weights_cuda.py tests/python/test_df_occupied_response_cuda.py tests/python/test_df_resident_response_cuda.py > .artifacts/issue394-000/v1/gpu-regression.log 2>&1
