#!/bin/sh
# Run from the refreshed source tree with its own CPU-only native library.
set -eu
: "${GENERATIVEQC_LIBRARY:?Set this source tree's own native CPU library}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/python:$PWD"
CPU=${CPU:-2}
taskset -c "$CPU" ctest --test-dir build/cpu-revalidation --output-on-failure -j1 \
  -R '^generativeqc_(dft|dft_density_source|uks|uks_state|ks_final_state|final_state|rks_response|uks_response|scf_diagnostic)_tests$'
set --
if [ -n "${JUNIT_XML:-}" ]; then set -- "--junitxml=$JUNIT_XML"; fi
taskset -c "$CPU" python -m pytest -q -rs "$@" \
  tests/python/test_cpu_ks_final_closure.py tests/python/test_ks_stage_baseline.py \
  tests/python/test_ks_diagnostics.py \
  tests/python/test_dft_complete_cpu.py::test_native_late_grid_failure_stale_lease_and_changed_geometry \
  tests/python/test_stationary_grid_schedule.py tests/python/test_stationary_geometry_resources.py \
  tests/python/test_stationary_large_grid_work.py tests/python/test_stationary_cuda_merge_boundary.py \
  tests/python/test_dft_mp_v1_capacity.py tests/python/test_reference_xc_backend.py \
  tests/python/test_pbe0_r2scan_integration.py tests/python/test_vwn_spin_boundary.py \
  tests/python/test_libxc_maple_vwn.py tests/python/test_libxc_maple_b88_binding.py
