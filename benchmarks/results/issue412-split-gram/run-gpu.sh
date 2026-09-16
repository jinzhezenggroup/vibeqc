#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
nvidia-smi --query-gpu=name,uuid,driver_version,temperature.gpu,pstate --format=csv > .artifacts/issue412/experiment-v1/device.csv
/tmp/vibeqc-pr-review-env/bin/python .artifacts/issue412/run-trials.py --output .artifacts/issue412/experiment-v1
for sanitizer in memcheck initcheck; do
  /group/software/cuda-12.9.1/bin/compute-sanitizer --tool "$sanitizer" --error-exitcode=1 \
    .artifacts/issue412/trial .artifacts/issue412/experiment-v1/independent-small-u.bin \
    13 63 1 ".artifacts/issue412/experiment-v1/sanitizer-$sanitizer" 5 2147483648 \
    > ".artifacts/issue412/experiment-v1/$sanitizer.log" 2>&1
done
