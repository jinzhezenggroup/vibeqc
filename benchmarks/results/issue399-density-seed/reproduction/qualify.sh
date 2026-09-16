#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VIBEQC_LIBRARY="$PWD/.artifacts/issue399/v1/libvibeqc.so"
export LD_LIBRARY_PATH="/group/software/cuda-12.9.1/lib64"
export VIBEQC_RESOURCE_CUDA_TEST=1
build/cuda-release-sm120/vibeqc_df_density_seed_tests
/group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck --error-exitcode 99 build/cuda-release-sm120/vibeqc_df_density_seed_tests
for test in vibeqc_density_fitting_tests vibeqc_cuda_diis_tests vibeqc_df_occupied_response_tests; do
  build/cuda-release-sm120/"$test"
done
export VIBEQC_DF_EXCHANGE=occupied VIBEQC_DF_FINAL_EXCHANGE=dense
for aos in 768 384; do
  if [[ "$aos" == 384 ]]; then case_name=water-hexadecamer-2s4-def2-svp-spherical; else case_name=water-32mer-4s4-def2-svp-spherical; fi
  checkpoint="/home/jzzeng/codes/vibeqc-issues388-391/.artifacts/issues388-391/final/ablations/$aos-VIBEQC_DF_DIIS_DOTS.checkpoint"
  reference="benchmarks/results/issue377-379-df/gpu4pyscf/$case_name.json"
  /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint --aos "$aos" --control VIBEQC_DF_SEED_EXCHANGE --policies dense factor --repeats 5 --expected-iterations 3 --components-after --skip-cold --warm-checkpoint-in "$checkpoint" --reference "$reference" --output ".artifacts/issue399/v1/$aos-seed.json"
  /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint --aos "$aos" --energy-only --control VIBEQC_DF_SEED_EXCHANGE --policies dense factor --repeats 5 --expected-iterations 3 --skip-cold --warm-checkpoint-in "$checkpoint" --reference "$reference" --output ".artifacts/issue399/v1/$aos-seed-energy.json"
  export VIBEQC_DF_SEED_EXCHANGE=factor
  /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint --aos "$aos" --control VIBEQC_DF_FINAL_EXCHANGE --policies dense occupied --repeats 5 --expected-iterations 3 --components-after --skip-cold --warm-checkpoint-in "$checkpoint" --reference "$reference" --output ".artifacts/issue399/v1/$aos-final.json"
  /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint --aos "$aos" --energy-only --control VIBEQC_DF_FINAL_EXCHANGE --policies dense occupied --repeats 5 --expected-iterations 3 --skip-cold --warm-checkpoint-in "$checkpoint" --reference "$reference" --output ".artifacts/issue399/v1/$aos-final-energy.json"
  VIBEQC_DF_SEED_VERIFY=1 /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint --aos "$aos" --control VIBEQC_DF_SEED_EXCHANGE --policies factor --repeats 1 --trace --journal --expected-iterations 3 --skip-cold --warm-checkpoint-in "$checkpoint" --reference "$reference" --output ".artifacts/issue399/v1/$aos-verification.json"
  unset VIBEQC_DF_SEED_EXCHANGE
done
