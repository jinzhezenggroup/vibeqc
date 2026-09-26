set -eu
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VIBEQC_LIBRARY="$PWD/.artifacts/issue408/verified/libvibeqc.so"
export LD_LIBRARY_PATH="/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
for aos in 96 192 384 768; do
  case "$aos" in
    96) name=water-tetramer-def2-svp-spherical ;;
    192) name=water-octamer-s4-def2-svp-spherical ;;
    384) name=water-hexadecamer-2s4-def2-svp-spherical ;;
    768) name=water-32mer-4s4-def2-svp-spherical ;;
  esac
  checkpoint=()
  if [ "$aos" -ge 384 ]; then
    checkpoint=(--skip-cold --expected-iterations 3 --warm-checkpoint-in "/home/jzzeng/codes/vibeqc-issues388-391/.artifacts/issues388-391/final/ablations/$aos-VIBEQC_DF_DIIS_DOTS.checkpoint")
  else
    checkpoint=(--warm-checkpoint-out "$PWD/.artifacts/issue408/verified/$aos.checkpoint")
  fi
  /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint --aos "$aos" --output ".artifacts/issue408/verified/$aos-forces.json" --repeats 5 --control VIBEQC_DF_REFERENCE_FINAL_VALIDATION --policies 1 0 --components-after --source-patch .artifacts/issue408/verified/source.patch "${checkpoint[@]}" --reference "benchmarks/results/issue377-379-df/gpu4pyscf/$name.json"
  if [ "$aos" -lt 384 ]; then
    checkpoint=(--skip-cold --warm-checkpoint-in "$PWD/.artifacts/issue408/verified/$aos.checkpoint")
  fi
  /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint --aos "$aos" --output ".artifacts/issue408/verified/$aos-energy.json" --repeats 5 --control VIBEQC_DF_REFERENCE_FINAL_VALIDATION --policies 1 0 --energy-only --components-after --source-patch .artifacts/issue408/verified/source.patch "${checkpoint[@]}" --reference "benchmarks/results/issue377-379-df/gpu4pyscf/$name.json"
done
