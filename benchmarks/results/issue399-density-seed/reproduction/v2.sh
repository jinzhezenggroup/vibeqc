#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VIBEQC_RESOURCE_CUDA_TEST=1
export VIBEQC_LIBRARY="$PWD/.artifacts/issue399/v2/libvibeqc.so"
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64
build/cuda-release-sm120/vibeqc_df_density_seed_tests > .artifacts/issue399/v2/native-test.log 2>&1
/group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck --error-exitcode 99 build/cuda-release-sm120/vibeqc_df_density_seed_tests > .artifacts/issue399/v2/memcheck.log 2>&1
/tmp/vibeqc-pr-review-env/bin/python .artifacts/issue399/v2-clean.py
/tmp/vibeqc-pr-review-env/bin/python -m pytest -q tests/python/test_df_exchange_selector_cuda.py > .artifacts/issue399/v2/selector.log 2>&1
bash .artifacts/issue399/profile.sh
