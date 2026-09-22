#!/usr/bin/env bash
set -u
: "${SLURM_JOB_ID:?GPU execution requires Slurm}"
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}
task_evidence=/home/jzzeng/codes/vibeqc-acceptance-evidence-20260922
for task_binary in diagnose_df_counters_v2 test_density_fitting_fixed_v2; do
  "$task_evidence/$task_binary" > "$task_evidence/$task_binary.log" 2>&1
  echo "$task_binary exit=$?"
done
cd /home/jzzeng/codes/vibeqc-acceptance-fixes-20260922 || exit 1
export PYTHONPATH=python:.
export VIBEQC_LIBRARY=/home/jzzeng/codes/vibeqc-acceptance-closeout-20260922/build-acceptance/libvibeqc.so
export VIBEQC_TEST_FOCK_DEVICE=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/jzzeng/codes/vibeqc-torch-boundaries/.venv/bin/python -m pytest -q \
 tests/python/test_fock.py::test_native_failure_publication_and_preflight \
 tests/python/test_calculator.py::test_cartesian_p_shell_energy_force_and_cuda_agreement \
 --junitxml="$task_evidence/hf-contracts-fixed-v2.xml" > "$task_evidence/hf-contracts-fixed-v2.log" 2>&1
echo "hf-contracts-fixed-v2 exit=$?"
