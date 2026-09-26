#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
root="$PWD/.artifacts/issues388-391"
out="$root/derivative-ablations"
mkdir -p "$out"
# Two 768-AO prepared owners do not fit in 32 GiB. Reconstruct sequentially
# in ABCCBA blocks, with two then three clean samples per variant (five total).
for aos in 768 384; do
  if [[ "$aos" == 384 ]]; then case_name=water-hexadecamer-2s4-def2-svp-spherical; else case_name=water-32mer-4s4-def2-svp-spherical; fi
  reference="benchmarks/results/issue377-379-df/gpu4pyscf/$case_name.json"
  for phase in forward reverse; do
    variants=(legacy shared center); repeats=2; components=(--components-after)
    if [[ "$phase" == reverse ]]; then variants=(center shared legacy); repeats=3; components=(); fi
    for variant in "${variants[@]}"; do
      export VIBEQC_LIBRARY="$root/derivative-$variant/libvibeqc.so"
      export LD_LIBRARY_PATH="$root/derivative-$variant:/group/software/cuda-12.9.1/lib64"
      /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint --aos "$aos" --policies auto --repeats "$repeats" "${components[@]}" --reference "$reference" --cold-control VIBEQC_DF_RESIDENT_EXCHANGE=legacy --cold-control VIBEQC_DF_DIIS_DOTS=serial --warm-checkpoint-out "$out/$aos-$phase-$variant.checkpoint" --output "$out/$aos-$phase-$variant.json"
    done
  done
done
