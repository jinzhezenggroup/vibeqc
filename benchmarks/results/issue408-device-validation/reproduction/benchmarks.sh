#!/usr/bin/env bash
# Run inside the finite Slurm allocation shown in the evidence README.
set -eu
: "${SLURM_JOB_ID:?Run this script through srun on the main GPU partition}"
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VIBEQC_LIBRARY="${VIBEQC_LIBRARY:-$PWD/build/cuda/libvibeqc.so}"
python_bin="${PYTHON:-python3}"
output="${OUTPUT:-.artifacts/issue408-reproduction}"
mkdir -p "$output"
for aos in 96 192 384 768; do
  case "$aos" in
    96) name=water-tetramer-def2-svp-spherical ;;
    192) name=water-octamer-s4-def2-svp-spherical ;;
    384) name=water-hexadecamer-2s4-def2-svp-spherical ;;
    768) name=water-32mer-4s4-def2-svp-spherical ;;
  esac
  iterations=3
  if [ "$aos" -eq 96 ]; then iterations=2; fi
  checkpoint="${CHECKPOINT_DIR:-$output}/$aos.checkpoint"
  if [ -f "$checkpoint" ]; then
    seed=(--skip-cold --warm-checkpoint-in "$checkpoint")
  else
    # A fresh seed reproduces the protocol, not the archived density hash.
    seed=(--warm-checkpoint-out "$checkpoint")
  fi
  "$python_bin" -m benchmarks.df_policy_endpoint --aos "$aos" \
    --output "$output/$aos-forces.json" --repeats 5 --expected-iterations "$iterations" \
    --control VIBEQC_DF_REFERENCE_FINAL_VALIDATION --policies 1 0 \
    --components-after "${seed[@]}" \
    --reference "benchmarks/results/issue377-379-df/gpu4pyscf/$name.json"
  "$python_bin" -m benchmarks.df_policy_endpoint --aos "$aos" \
    --output "$output/$aos-energy.json" --repeats 5 --energy-only --expected-iterations "$iterations" \
    --control VIBEQC_DF_REFERENCE_FINAL_VALIDATION --policies 1 0 \
    --components-after --skip-cold --warm-checkpoint-in "$checkpoint" \
    --reference "benchmarks/results/issue377-379-df/gpu4pyscf/$name.json"
done
