#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
root="$PWD/.artifacts/issues388-391"
out="$root/final/ablations"
mkdir -p "$out"
export VIBEQC_LIBRARY="$root/final/libvibeqc.so"
export LD_LIBRARY_PATH="$root/final:/group/software/cuda-12.9.1/lib64"
for test in vibeqc_density_fitting_tests vibeqc_df_occupied_response_tests vibeqc_df_shell_pairs_tests vibeqc_cuda_diis_tests; do build/cuda-release-sm120/"$test"; done
for aos in 768 384; do
  if [[ "$aos" == 384 ]]; then case_name=water-hexadecamer-2s4-def2-svp-spherical; else case_name=water-32mer-4s4-def2-svp-spherical; fi
  reference="benchmarks/results/issue377-379-df/gpu4pyscf/$case_name.json"
  for control in VIBEQC_DF_RESIDENT_EXCHANGE VIBEQC_DF_RAW_REUSE VIBEQC_DF_RESPONSE_BATCHING VIBEQC_DF_DIIS_DOTS; do
    policies=(off auto)
    if [[ "$control" == VIBEQC_DF_RESIDENT_EXCHANGE ]]; then policies=(legacy flat full auto); export VIBEQC_DF_RAW_REUSE=off; fi
    if [[ "$control" == VIBEQC_DF_DIIS_DOTS ]]; then policies=(serial auto); fi
    /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint --aos "$aos" --control "$control" --policies "${policies[@]}" --repeats 5 --components-after --reference "$reference" --cold-control VIBEQC_DF_RESIDENT_EXCHANGE=legacy --cold-control VIBEQC_DF_DIIS_DOTS=serial --warm-checkpoint-out "$out/$aos-$control.checkpoint" --output "$out/$aos-$control.json"
    unset VIBEQC_DF_RAW_REUSE
  done
  for control in VIBEQC_DF_RESIDENT_EXCHANGE VIBEQC_DF_DIIS_DOTS; do
    if [[ "$control" == VIBEQC_DF_RESIDENT_EXCHANGE ]]; then policies=(legacy auto); else policies=(serial auto); fi
    /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint --aos "$aos" --energy-only --expected-iterations 3 --control "$control" --policies "${policies[@]}" --repeats 5 --reference "$reference" --cold-control VIBEQC_DF_RESIDENT_EXCHANGE=legacy --cold-control VIBEQC_DF_DIIS_DOTS=serial --warm-checkpoint-out "$out/$aos-$control-energy.checkpoint" --output "$out/$aos-$control-energy.json"
  done
done
/tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint --aos 384 --control VIBEQC_DF_RESPONSE_STORAGE --policies panel jk-scratch --repeats 5 --components-after --reference benchmarks/results/issue377-379-df/gpu4pyscf/water-hexadecamer-2s4-def2-svp-spherical.json --cold-control VIBEQC_DF_RESIDENT_EXCHANGE=legacy --cold-control VIBEQC_DF_DIIS_DOTS=serial --warm-checkpoint-out "$out/384-storage.checkpoint" --output "$out/384-VIBEQC_DF_RESPONSE_STORAGE.json"

for aos in 384 768; do
  if [[ "$aos" == 384 ]]; then case_name=water-hexadecamer-2s4-def2-svp-spherical; else case_name=water-32mer-4s4-def2-svp-spherical; fi
  /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint --aos "$aos" --policies auto --repeats 5 --components-after --reference "benchmarks/results/issue377-379-df/gpu4pyscf/$case_name.json" --warm-checkpoint-out "$root/final/$aos-normal.checkpoint" --output "$root/final/$aos-normal.json"
done
