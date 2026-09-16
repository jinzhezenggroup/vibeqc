#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a finite Slurm allocation}"
output_root="$PWD/.artifacts/issue412/validation-followup"
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export VIBEQC_LIBRARY="$PWD/.artifacts/issue412/candidate/libvibeqc.so"
export VIBEQC_RESOURCE_CUDA_TEST=1 VIBEQC_DF_RESIDENT_EXCHANGE=split4
printf '%s\n' "$SLURM_JOB_ID" > "$output_root/slurm-job.txt"
/tmp/vibeqc-pr-review-env/bin/python -m pytest -q -ra \
  'tests/python/test_df_final_projection_cuda.py::test_final_projection_replay_and_geometry[uhf]' \
  --junitxml="$output_root/pytest.xml" > "$output_root/pytest.log" 2>&1
for sanitizer in memcheck initcheck; do
  extra=()
  if [[ "$sanitizer" == memcheck ]]; then extra+=(--leak-check full); fi
  /group/software/cuda-12.9.1/bin/compute-sanitizer --tool "$sanitizer" --error-exitcode=1 "${extra[@]}" \
    ./build/cuda/vibeqc_density_fitting_tests > "$output_root/$sanitizer.log" 2>&1
done
