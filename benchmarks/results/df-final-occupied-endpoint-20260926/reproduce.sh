#!/usr/bin/env bash
# Build the matching sm_120 Release library before entering this Slurm job.
set -euo pipefail
: "${SLURM_JOB_ID:?Use srun --partition=main --gres=gpu:5090:1 with a finite --time}"
: "${VIBEQC_LIBRARY:?Set the exact built libvibeqc.so path}"
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="$PWD/python:$PWD"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
py=${VIBEQC_PYTHON:-python}
out=${1:-.artifacts/benchmarks/df-final-occupied-$SLURM_JOB_ID}
# Independent references and native endpoints run in separate processes so
# neither retains GPU allocations or runtime state belonging to the other.
for aos in 384 768; do
  for route in direct df; do
    "$py" -m benchmarks.compare_df_direct_endpoint reference --aos "$aos" --route "$route" \
      --output "$out/$aos/reference-$route"
  done
done
for aos in 384 768; do
  "$py" -m benchmarks.compare_df_direct_endpoint native --aos "$aos" --route direct \
    --repeats 5 --reference "$out/$aos/reference-direct/results.json" --output "$out/$aos/direct"
  "$py" -m benchmarks.compare_df_direct_endpoint native --aos "$aos" --route df --interleave \
    --repeats 5 --reference "$out/$aos/reference-df/results.json" --output "$out/$aos/df"
  for route in df-baseline df-final-k; do
    "$py" -m benchmarks.compare_df_direct_endpoint native --aos "$aos" --route "$route" \
      --repeats 1 --reference "$out/$aos/reference-df/results.json" --output "$out/$aos/$route"
  done
done
