#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run through a finite Slurm GPU allocation}"
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VIBEQC_LIBRARY="$PWD/build/cuda-release-sm120/libvibeqc.so"
export LD_LIBRARY_PATH="$PWD/build/cuda-release-sm120:/group/software/cuda-12.9.1/lib64"
export VIBEQC_RESOURCE_CUDA_TEST=1
/tmp/vibeqc-pr-review-env/bin/python - <<'PY'
import ctypes
from pathlib import Path
from vibeqc import _native
from vibeqc.autotune import source_identity
library = _native.load_library()
library.vibeqc_get_source_identity.restype = ctypes.c_char_p
actual = library.vibeqc_get_source_identity().decode()
assert actual == source_identity(Path.cwd()), (actual, source_identity(Path.cwd()))
print('Matched runtime source identity:', actual, flush=True)
PY
build/cuda-release-sm120/vibeqc_df_shell_pairs_tests
/group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  build/cuda-release-sm120/vibeqc_df_shell_pairs_tests
/tmp/vibeqc-pr-review-env/bin/python -m pytest -q -x \
  tests/python/test_df_shell_derivatives_cuda.py \
  tests/python/test_df_resident_response_cuda.py \
  tests/python/test_df_occupied_response_cuda.py
