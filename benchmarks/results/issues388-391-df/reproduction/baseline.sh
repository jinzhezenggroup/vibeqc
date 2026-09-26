#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VIBEQC_LIBRARY="$PWD/.artifacts/issues388-391/baseline/libvibeqc.so"
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64
for aos in 384 768; do
  if [[ "$aos" == 384 ]]; then case_name=water-hexadecamer-2s4-def2-svp-spherical; else case_name=water-32mer-4s4-def2-svp-spherical; fi
  reference="benchmarks/results/issue377-379-df/gpu4pyscf/$case_name.json"
  /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint --aos "$aos" --policies auto --repeats 5 --reference "$reference" --output ".artifacts/issues388-391/baseline/$aos-clean.json"
  VIBEQC_DF_SHELL_COUNTERS=1 /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint --aos "$aos" --policies auto --repeats 1 --trace --reference "$reference" --output ".artifacts/issues388-391/baseline/$aos-trace.json"
done
