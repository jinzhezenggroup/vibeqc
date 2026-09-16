set -eu
export PYTHONPATH=python:.
export VIBEQC_RESOURCE_CUDA_TEST=1
export VIBEQC_LIBRARY="$PWD/.artifacts/issue408/verified/libvibeqc.so"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
/group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck --leak-check full --error-exitcode=1 build/cuda/vibeqc_df_final_validation_tests
/group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck --leak-check full --error-exitcode=1 build/cuda/vibeqc_df_final_snapshot_tests
/group/software/cuda-12.9.1/bin/compute-sanitizer --tool initcheck --error-exitcode=1 build/cuda/vibeqc_df_final_validation_tests
/tmp/vibeqc-pr-review-env/bin/python -m pytest -q tests/python/test_fock_eigen_cuda.py tests/python/test_cuda_runtime.py tests/python/test_density_cuda.py tests/python/test_one_electron_derivatives_cuda.py
