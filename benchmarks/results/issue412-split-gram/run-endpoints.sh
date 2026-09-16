#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a finite Slurm allocation}"
output_root="$PWD/.artifacts/issue412/endpoints-v1"
if [[ -e "$output_root" ]]; then
  echo 'Refusing to overwrite endpoint evidence' >&2
  exit 1
fi
mkdir -p "$output_root"
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export VIBEQC_LIBRARY="$PWD/.artifacts/issue412/candidate/libvibeqc.so"
export VIBEQC_DF_REFERENCE_FINAL_VALIDATION=0 VIBEQC_DF_FINAL_PROJECTION=auto
export VIBEQC_DF_FORCE_SCREEN_ABS=off VIBEQC_DF_EXCHANGE=auto
export VIBEQC_DF_SEED_EXCHANGE=auto VIBEQC_DF_FINAL_EXCHANGE=auto
export VIBEQC_DF_SHELL_POLICY=auto
for aos in 384 768 96 192; do
  case "$aos" in
    96) case_name=water-tetramer-def2-svp-spherical ;;
    192) case_name=water-octamer-s4-def2-svp-spherical ;;
    384) case_name=water-hexadecamer-2s4-def2-svp-spherical ;;
    768) case_name=water-32mer-4s4-def2-svp-spherical ;;
  esac
  extra=()
  if [[ "$aos" -ge 384 ]]; then
    checkpoint="/home/jzzeng/codes/vibeqc-issues388-391/.artifacts/issues388-391/final/ablations/$aos-VIBEQC_DF_DIIS_DOTS.checkpoint"
    extra+=(--expected-iterations 3)
  else
    checkpoint="/home/jzzeng/codes/vibeqc-issue408/.artifacts/issue408/verified/$aos.checkpoint"
  fi
  for observable in forces energy; do
    properties=()
    if [[ "$observable" == energy ]]; then properties+=(--energy-only); fi
    /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint \
      --aos "$aos" --output "$output_root/$aos-$observable.json" --repeats 7 \
      --control VIBEQC_DF_RESIDENT_EXCHANGE --policies auto split4 --components-after \
      --source-patch .artifacts/issue412/candidate/source.patch --skip-cold \
      --warm-checkpoint-in "$checkpoint" "${extra[@]}" "${properties[@]}" \
      --reference "benchmarks/results/issue377-379-df/gpu4pyscf/$case_name.json"
  done
done
