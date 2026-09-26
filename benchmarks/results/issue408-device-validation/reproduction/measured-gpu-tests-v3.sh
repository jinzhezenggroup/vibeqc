set -eu
export PYTHONPATH=python:.
export VIBEQC_RESOURCE_CUDA_TEST=1
export VIBEQC_LIBRARY="$PWD/.artifacts/issue408/verified/libvibeqc.so"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=name,driver_version,memory.total,uuid --format=csv,noheader
ctest --test-dir build/cuda --output-on-failure -R 'vibeqc_(df_final_validation|df_final_snapshot|df_eigensystem|final_state|eigen_frame|cuda_reference_export|cuda_fock_composition|df_occupied_response|df_density_seed)_tests'
/tmp/vibeqc-pr-review-env/bin/python -m pytest -q tests/python/test_df_final_state_cuda.py tests/python/test_df_force_state_export_cuda.py tests/python/test_df_final_eigen_cuda.py tests/python/test_df_property_budget_cuda.py tests/python/test_df_retry_eigen_cuda.py tests/python/test_df_setup_eigen_cuda.py tests/python/test_hf_resources_cuda.py tests/python/test_df_final_projection_cuda.py

/group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck --leak-check full --error-exitcode=1 build/cuda/vibeqc_df_final_snapshot_tests
/group/software/cuda-12.9.1/bin/compute-sanitizer --tool initcheck --error-exitcode=1 build/cuda/vibeqc_df_final_snapshot_tests
