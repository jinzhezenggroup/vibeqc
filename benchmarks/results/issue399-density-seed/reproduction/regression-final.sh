#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VIBEQC_RESOURCE_CUDA_TEST=1
export VIBEQC_LIBRARY="$PWD/.artifacts/issue399/v2/libvibeqc.so"
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64
export VIBEQC_DF_EXCHANGE=occupied VIBEQC_DF_SEED_EXCHANGE=factor VIBEQC_DF_FINAL_EXCHANGE=occupied
/tmp/vibeqc-pr-review-env/bin/python -m pytest -q -x tests/python/test_df_occupied_cuda.py tests/python/test_df_occupied_response_cuda.py tests/python/test_df_resident_response_cuda.py tests/python/test_df_exchange_selector_cuda.py tests/python/test_cuda_density_fitting_source.py tests/python/test_df_retry_eigen_cuda.py
