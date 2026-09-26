set -eu
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VIBEQC_LIBRARY="$PWD/.artifacts/issue408/verified/libvibeqc.so"
export LD_LIBRARY_PATH="/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
for aos in 384 768; do
  /tmp/vibeqc-pr-review-env/bin/python .artifacts/issue408/warm-memory-v3.py --aos "$aos" --output ".artifacts/issue408/verified/$aos-warm-memory.json" --checkpoint "/home/jzzeng/codes/vibeqc-issues388-391/.artifacts/issues388-391/final/ablations/$aos-VIBEQC_DF_DIIS_DOTS.checkpoint"
done
