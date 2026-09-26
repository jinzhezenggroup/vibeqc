#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=python:.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}
export VIBEQC_LIBRARY=/home/jzzeng/codes/qc-mixed-precision/.artifacts/issue308-shell-block/v1/libvibeqc.so
unset VIBEQC_DF_TRACE VIBEQC_DF_HOST_TRACE VIBEQC_DF_PROGRESS_TRACE VIBEQC_DF_SHELL_COUNTERS
unset VIBEQC_DF_RESPONSE_UPLOAD_PROBE VIBEQC_DF_RESPONSE_SCATTER_PROBE
issue308_artifacts=/home/jzzeng/codes/qc-mixed-precision/.artifacts/issue308-shell-block/v1
cd "$issue308_artifacts/source"
python3 - <<'PY'
import json,os,pathlib
from vibeqc.autotune import source_identity
p=pathlib.Path(os.environ['VIBEQC_LIBRARY']).parent
assert source_identity(pathlib.Path.cwd()) == json.loads((p/'provenance.json').read_text())['source_identity']
(p/'clean-slurm-job.txt').write_text(os.environ['SLURM_JOB_ID']+'\n')
PY
nvidia-smi --query-gpu=name,uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv > "$issue308_artifacts/clean-gpu-before.csv"
for batch in 1 4; do
  for aos in 192 384; do
    for route in generic shell-sp; do
      export VIBEQC_DF_WEIGHTED_EXECUTION=$route
      /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.issue308_response_timeline \
        --aos "$aos" --batch "$batch" --repeats 2 --library "$VIBEQC_LIBRARY" \
        --output "$issue308_artifacts/${aos}ao-b${batch}-${route}" \
        > "$issue308_artifacts/${aos}ao-b${batch}-${route}.log" 2>&1
    done
  done
done
nvidia-smi --query-gpu=name,uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv > "$issue308_artifacts/clean-gpu-after.csv"
