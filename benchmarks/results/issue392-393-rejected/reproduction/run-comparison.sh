#!/usr/bin/env bash
# Capture new evidence from frozen libraries; never build during timing.
# Stages: checkpoint, clean (one candidate), clean-early (direct + register).
set -euo pipefail
: "${SLURM_JOB_ID:?Run through a finite Slurm main allocation with one 5090 GPU}"
: "${VIBEQC_WORK_BASELINE:?Frozen baseline directory}"
: "${VIBEQC_WORK_OUTPUT:?Output directory for new evidence}"
: "${VIBEQC_WORK_CHECKPOINTS:?Shared checkpoint directory}"
work_python=${VIBEQC_WORK_PYTHON:-python}
work_cuda=${VIBEQC_WORK_CUDA:-/group/software/cuda-12.9.1}
stage=${1:?checkpoint, clean, or clean-early}
aos=${2:?384 or 768}
case "$stage" in
  checkpoint) groups=(baseline-1) ;;
  clean)
    : "${VIBEQC_WORK_CANDIDATE:?Frozen candidate directory}"
    groups=(baseline-2 candidate-2 candidate-3 baseline-3)
    ;;
  clean-early)
    : "${VIBEQC_WORK_DIRECT:?Frozen direct-folding directory}"
    : "${VIBEQC_WORK_REGISTER:?Frozen register-folding directory}"
    groups=(baseline-2 direct-2 register-2 register-3 direct-3 baseline-3)
    ;;
  *) exit 2 ;;
esac
case "$aos" in
  384) case_name=water-hexadecamer-2s4 ;;
  768) case_name=water-32mer-4s4 ;;
  *) exit 2 ;;
esac
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
reference="benchmarks/results/issue377-379-df/gpu4pyscf/$case_name-def2-svp-spherical.json"
checkpoint="$VIBEQC_WORK_CHECKPOINTS/$aos-normal.checkpoint"
[[ -r "$reference" ]] || { echo "Run from the repository root" >&2; exit 2; }
if [[ "$stage" != checkpoint && ! -r "$checkpoint" ]]; then
  echo "Create the shared baseline checkpoint first: $checkpoint" >&2
  exit 2
fi
mkdir -p "$VIBEQC_WORK_OUTPUT" "$VIBEQC_WORK_CHECKPOINTS"
common=(--aos "$aos" --control VIBEQC_DF_SHELL_WORK --reference "$reference" --policies 0)
for group in "${groups[@]}"; do
  case "${group%-*}" in
    baseline) library_dir=$VIBEQC_WORK_BASELINE ;;
    candidate) library_dir=$VIBEQC_WORK_CANDIDATE ;;
    direct) library_dir=$VIBEQC_WORK_DIRECT ;;
    register) library_dir=$VIBEQC_WORK_REGISTER ;;
  esac
  # Pin the binary and its own source patch, rather than the current checkout.
  for input in libvibeqc.so source.patch; do
    [[ -r "$library_dir/$input" ]] || { echo "Missing $library_dir/$input" >&2; exit 2; }
  done
  export VIBEQC_LIBRARY="$library_dir/libvibeqc.so"
  export LD_LIBRARY_PATH="$library_dir:$work_cuda/lib64"
  if [[ "$stage" == checkpoint ]]; then
    "$work_python" -m benchmarks.df_policy_endpoint "${common[@]}" \
      --source-patch "$library_dir/source.patch" --repeats 1 \
      --warm-checkpoint-out "$checkpoint" \
      --output "$VIBEQC_WORK_OUTPUT/$aos-checkpoint.json"
  else
    # Clean samples precede the separate component pass. The endpoint runner
    # disables instrumentation during clean timing and refuses output reuse.
    "$work_python" -m benchmarks.df_policy_endpoint "${common[@]}" \
      --source-patch "$library_dir/source.patch" --repeats "${group##*-}" \
      --expected-iterations 3 --components-after \
      --warm-checkpoint-in "$checkpoint" --skip-cold \
      --output "$VIBEQC_WORK_OUTPUT/$aos-$group.json"
  fi
done
