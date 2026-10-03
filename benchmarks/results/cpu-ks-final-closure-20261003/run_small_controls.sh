#!/bin/sh
# Run from the exact source tree whose own library is selected below.
set -eu
: "${GENERATIVEQC_LIBRARY:?Set the library from this source tree's CPU build}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/python:$PWD"
CPU=${CPU:-2}
taskset -c "$CPU" ctest --test-dir build/cpu-revalidation --output-on-failure -j1 \
  -R '^generativeqc_(dft|dft_density_source|uks|uks_state|ks_final_state|final_state|rks_response|uks_response|scf_diagnostic)_tests$'
taskset -c "$CPU" python -m pytest -q -rs \
  tests/python/test_ks_stage_baseline.py tests/python/test_ks_diagnostics.py \
  tests/python/test_cpu_ks_final_closure.py tests/python/test_ks_snapshot_provider_proof.py \
  tests/python/test_snapshot_grid_cache.py tests/python/test_response_native_rks.py \
  tests/python/test_response_native_uks.py tests/python/test_pbe0_r2scan_integration.py \
  tests/python/test_vv10_self_consistent.py tests/python/test_stationary_rsh_cpu.py \
  'tests/python/test_dft_complete_cpu.py::test_complete_asymmetric_water_analytic_and_reconverged_fd[pbe-rks-native]' \
  'tests/python/test_dft_complete_cpu.py::test_complete_open_shell_uks_analytic_and_reconverged_fd[pbe-uks-native]' \
  tests/python/test_dft_complete_cpu.py::test_failure_isolation_native_malformed_geometry_and_detached_state \
  tests/python/test_dft_complete_cpu.py::test_native_late_grid_failure_stale_lease_and_changed_geometry
