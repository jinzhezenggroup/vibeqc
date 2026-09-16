#!/usr/bin/env bash
set -euo pipefail
capture_root="$PWD/.artifacts/issue412/captured-v2"
mkdir -p "$capture_root"
export LD_PRELOAD="$PWD/.artifacts/issue412/capture-v2.so"
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/jzzeng/codes/vibeqc-issue394-batch
export VIBEQC_LIBRARY="$PWD/.artifacts/issue404/baseline/libvibeqc.so"
export VIBEQC_DF_REFERENCE_FINAL_VALIDATION=0
export VIBEQC_DF_FINAL_PROJECTION=auto
export VIBEQC_DF_FORCE_SCREEN_ABS=off
for aos in 384 768; do
  mkdir -p "$capture_root/$aos"
  if [ "$aos" = 384 ]; then
    export VIBEQC_DF_EXCHANGE=occupied VIBEQC_DF_SEED_EXCHANGE=factor VIBEQC_DF_FINAL_EXCHANGE=occupied
  else
    export VIBEQC_DF_EXCHANGE=auto VIBEQC_DF_SEED_EXCHANGE=auto VIBEQC_DF_FINAL_EXCHANGE=auto
  fi
  export VIBEQC_GRAM_CAPTURE_N="$aos"
  export VIBEQC_GRAM_CAPTURE_DIR="$capture_root/$aos"
  export VIBEQC_DF_TRACE="$capture_root/$aos/capture-trace.jsonl"
  export VIBEQC_DF_PROGRESS_TRACE="$capture_root/$aos/capture-journal.jsonl"
  case "$aos" in
    384) case_name=water-hexadecamer-2s4-def2-svp-spherical ;;
    768) case_name=water-32mer-4s4-def2-svp-spherical ;;
  esac
  /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint \
    --aos "$aos" --output "$capture_root/$aos/endpoint.json" --repeats 1 \
    --control VIBEQC_DF_SHELL_POLICY --policies auto --trace --energy-only \
    --source-patch .artifacts/issue404/baseline/source.patch --skip-cold --expected-iterations 3 \
    --warm-checkpoint-in "/home/jzzeng/codes/vibeqc-issues388-391/.artifacts/issues388-391/final/ablations/$aos-VIBEQC_DF_DIIS_DOTS.checkpoint" \
    --reference "benchmarks/results/issue377-379-df/gpu4pyscf/$case_name.json"
done
