#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run through srun on main with --gres=gpu:5090:1 and a finite --time}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign GPU visibility; preserve its device selection}"
cd /home/jzzeng/codes/vibeqc-issue-189
export PYTHONPATH=python:.
export VIBEQC_LIBRARY=$PWD/build-cuda/libvibeqc.so
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}
projection_python=/home/jzzeng/codes/vibeqc-issue-204/.venv/bin/python
printf 'Slurm job: %s; CUDA_VISIBLE_DEVICES: %s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$projection_python" - <<'PY'
import ctypes
from pathlib import Path
from vibeqc.autotune import source_identity
from vibeqc import Calculator
library = Calculator(device='cuda')._library
library.vibeqc_get_source_identity.restype = ctypes.c_char_p
identity = library.vibeqc_get_source_identity().decode()
assert identity == source_identity(Path.cwd())
print('Verified source identity:', identity)
PY
ctest --test-dir build-cuda --output-on-failure -E vibeqc_aot_profile_tests
VIBEQC_PROJECTION_CUDA_TEST=1 "$projection_python" -m pytest tests/python/test_basis_projection_cuda.py -q --junitxml=/tmp/issue189-gpu-projection.xml
VIBEQC_CHECKPOINT_DEVICE=cuda "$projection_python" -m pytest tests/python/test_checkpoint.py -q --junitxml=/tmp/issue189-gpu-checkpoint.xml
for projection_case in h2-rhf-small-large h2-uhf-small-large h2-rhf-same h2-rhf-large-small; do
  "$projection_python" benchmarks/basis_projection_gate.py --device cuda --case "$projection_case" --repeats 5 --output "/tmp/issue189-bench/${projection_case}-cuda.json"
done
