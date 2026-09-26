#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run through a finite Slurm GPU allocation}"
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
baseline=/home/jzzeng/codes/vibeqc-issues388-391/.artifacts/issues388-391/final
candidate="$PWD/.artifacts/issues392-395/work-v1"
out="$candidate/qualification"
mkdir -p "$out"
# Fixed-density ABBA groups retain five samples per binary. Complete all builds
# before scheduling this script; neither counters nor traces enter these samples.
for aos in 384 768; do
  if [[ "$aos" == 384 ]]; then name=water-hexadecamer-2s4; else name=water-32mer-4s4; fi
  reference="benchmarks/results/issue377-379-df/gpu4pyscf/$name-def2-svp-spherical.json"
  checkpoint="$baseline/$aos-normal.checkpoint"
  for group in baseline-2 candidate-2 candidate-3 baseline-3; do
    selected=${group%-*}
    repeats=${group#*-}
    if [[ "$selected" == baseline ]]; then library_dir=$baseline; else library_dir=$candidate; fi
    export VIBEQC_LIBRARY="$library_dir/libvibeqc.so"
    export LD_LIBRARY_PATH="$library_dir:/group/software/cuda-12.9.1/lib64"
    /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint \
      --aos "$aos" --control VIBEQC_DF_SHELL_WORK --policies 0 \
      --repeats "$repeats" --expected-iterations 3 --components-after \
      --reference "$reference" --warm-checkpoint-in "$checkpoint" --skip-cold \
      --output "$out/$aos-$group.json"
  done
  export VIBEQC_LIBRARY="$candidate/libvibeqc.so"
  export LD_LIBRARY_PATH="$candidate:/group/software/cuda-12.9.1/lib64"
  export VIBEQC_DF_SHELL_COUNTERS=1
  /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint \
    --aos "$aos" --control VIBEQC_DF_SHELL_WORK --policies 0 1 --repeats 1 \
    --expected-iterations 3 --trace --reference "$reference" \
    --warm-checkpoint-in "$checkpoint" --skip-cold --output "$out/$aos-overhead.json"
  /group/software/cuda-12.9.1/bin/nsys profile --trace=cuda,nvtx --sample=none \
    --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi \
    --capture-range-end=repeat:2 --export=sqlite --output "$out/$aos-work" \
    /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint \
    --aos "$aos" --control VIBEQC_DF_SHELL_WORK --policies 0 1 --repeats 1 \
    --expected-iterations 3 --trace --cuda-profile --reference "$reference" \
    --warm-checkpoint-in "$checkpoint" --skip-cold --output "$out/$aos-profile.json"
  /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_shell_work_ledger \
    --trace "$out/$aos-profile.0-1.jsonl" --measurement "$out/$aos-profile.json" \
    --generated-header "$candidate/generated_df_shell_derivatives.cuh" \
    --nsys "$out/$aos-work.2.sqlite" --output "$out/$aos-work-ledger.json"
  unset VIBEQC_DF_SHELL_COUNTERS
done
