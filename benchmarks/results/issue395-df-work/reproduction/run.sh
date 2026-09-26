#!/usr/bin/env bash
# Run one finite Slurm stage at a time after both libraries have been built.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm main with one 5090 GPU}"
: "${VIBEQC_WORK_BASELINE:?Directory containing baseline libvibeqc.so and source.patch}"
: "${VIBEQC_WORK_CANDIDATE:?Directory containing candidate libvibeqc.so and source.patch}"
: "${VIBEQC_WORK_OUTPUT:?Fresh output directory}"
: "${VIBEQC_WORK_CHECKPOINTS:?Checkpoint directory shared by all stages}"
work_python=${VIBEQC_WORK_PYTHON:-python}
work_cuda=${VIBEQC_WORK_CUDA:-/group/software/cuda-12.9.1}
stage=${1:?checkpoint, clean, or profile}
aos=${2:?384 or 768}
case "$aos" in
  384) case_name=water-hexadecamer-2s4 ;;
  768) case_name=water-32mer-4s4 ;;
  *) exit 2 ;;
esac
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
reference="benchmarks/results/issue377-379-df/gpu4pyscf/$case_name-def2-svp-spherical.json"
checkpoint="$VIBEQC_WORK_CHECKPOINTS/$aos-normal.checkpoint"
mkdir -p "$VIBEQC_WORK_OUTPUT" "$VIBEQC_WORK_CHECKPOINTS"
select_library() {
  export VIBEQC_LIBRARY="$1/libvibeqc.so"
  export LD_LIBRARY_PATH="$1:$work_cuda/lib64"
}
common=(--aos "$aos" --control VIBEQC_DF_SHELL_WORK --reference "$reference")
case "$stage" in
  checkpoint)
    select_library "$VIBEQC_WORK_BASELINE"
    "$work_python" -m benchmarks.df_policy_endpoint "${common[@]}" \
      --policies 0 --repeats 1 --warm-checkpoint-out "$checkpoint" \
      --output "$VIBEQC_WORK_OUTPUT/$aos-checkpoint.json"
    ;;
  clean)
    # The runner strips traces/counters from these clean endpoint samples and
    # enables them only for the separate --components-after diagnostic pass.
    for group in baseline-2 candidate-2 candidate-3 baseline-3; do
      if [[ "$group" == baseline-* ]]; then
        select_library "$VIBEQC_WORK_BASELINE"
      else
        select_library "$VIBEQC_WORK_CANDIDATE"
      fi
      "$work_python" -m benchmarks.df_policy_endpoint "${common[@]}" \
        --policies 0 --repeats "${group#*-}" --expected-iterations 3 \
        --components-after --warm-checkpoint-in "$checkpoint" --skip-cold \
        --output "$VIBEQC_WORK_OUTPUT/$aos-$group.json"
    done
    ;;
  profile)
    select_library "$VIBEQC_WORK_CANDIDATE"
    export VIBEQC_DF_SHELL_COUNTERS=1
    options=("${common[@]}" --policies 0 1 --repeats 1 --expected-iterations 3
      --trace --warm-checkpoint-in "$checkpoint" --skip-cold)
    "$work_python" -m benchmarks.df_policy_endpoint "${options[@]}" \
      --output "$VIBEQC_WORK_OUTPUT/$aos-overhead.json"
    "$work_cuda/bin/nsys" profile --trace=cuda,nvtx --sample=none \
      --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi \
      --capture-range-end=repeat:2 --export=sqlite \
      --output "$VIBEQC_WORK_OUTPUT/$aos-work" \
      "$work_python" -m benchmarks.df_policy_endpoint "${options[@]}" \
      --cuda-profile --output "$VIBEQC_WORK_OUTPUT/$aos-profile.json"
    "$work_python" -m benchmarks.df_shell_work_ledger \
      --trace "$VIBEQC_WORK_OUTPUT/$aos-profile.0-1.jsonl" \
      --measurement "$VIBEQC_WORK_OUTPUT/$aos-profile.json" \
      --generated-header "$VIBEQC_WORK_CANDIDATE/generated_df_shell_derivatives.cuh" \
      --nsys "$VIBEQC_WORK_OUTPUT/$aos-work.2.sqlite" \
      --output "$VIBEQC_WORK_OUTPUT/$aos-work-ledger.json"
    ;;
  *) exit 2 ;;
esac
