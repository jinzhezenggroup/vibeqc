set -eu
export PYTHONPATH=python:. PYTHONUNBUFFERED=1
export VIBEQC_ONE_ELECTRON_DERIVATIVE_CUDA_TEST=1
export VIBEQC_LIBRARY="$PWD/.artifacts/issue408/verified/libvibeqc.so"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH="/group/software/cuda-12.9.1/bin:$PATH"
export LD_LIBRARY_PATH="/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
/tmp/vibeqc-pr-review-env/bin/python -m pytest -q tests/python/test_one_electron_derivatives_cuda.py --durations=8
