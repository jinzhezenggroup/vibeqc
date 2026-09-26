#!/usr/bin/env bash
set -euo pipefail
root=/inspire/qb-ilm/project/chemicalreaction/diwenxi-CZXS25120072/vibeqc-workspace
source "$root/activate-311.sh"
task="$root/build-171/fprojector-20260918"
trap 'echo $? > "$task/evidence/cpu-input-checks.exit"' EXIT
cd "$task/candidate"
export PYTHONPATH="$PWD/python" VIBEQC_LIBRARY="$task/cpu/libvibeqc.so"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest tests/python/test_external_basis.py -q > "$task/evidence/cpu-input-checks.log" 2>&1
