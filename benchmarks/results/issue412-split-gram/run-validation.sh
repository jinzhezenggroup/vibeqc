#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a finite Slurm allocation}"
output_root="$PWD/.artifacts/issue412/validation-v1"
if [[ -e "$output_root" ]]; then
  echo 'Refusing to overwrite validation evidence' >&2
  exit 1
fi
mkdir -p "$output_root"
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export VIBEQC_LIBRARY="$PWD/.artifacts/issue412/candidate/libvibeqc.so"
export VIBEQC_RESOURCE_CUDA_TEST=1 VIBEQC_DF_RESIDENT_EXCHANGE=split4
/tmp/vibeqc-pr-review-env/bin/python - "$output_root/job.json" <<'META'
import ctypes, hashlib, json, os, sys
from pathlib import Path
lib=Path(os.environ["VIBEQC_LIBRARY"])
native=ctypes.CDLL(str(lib))
native.vibeqc_get_source_identity.restype=ctypes.c_char_p
Path(sys.argv[1]).write_text(json.dumps({
    "slurm_job_id":os.environ["SLURM_JOB_ID"],
    "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
    "library_sha256":hashlib.sha256(lib.read_bytes()).hexdigest(),
    "native_source_identity":native.vibeqc_get_source_identity().decode(),
},indent=2)+"\n")
META
nvidia-smi --query-gpu=name,uuid,driver_version,temperature.gpu,pstate --format=csv > "$output_root/device.csv"
export VIBEQC_DF_TRACE="$output_root/native-trace.jsonl"
./build/cuda/vibeqc_density_fitting_tests > "$output_root/native.log" 2>&1
unset VIBEQC_DF_TRACE
/tmp/vibeqc-pr-review-env/bin/python -m pytest -q -ra \
  tests/python/test_df_occupied_cuda.py \
  tests/python/test_df_final_projection_cuda.py \
  tests/python/test_df_occupied_response_cuda.py \
  tests/python/test_df_resident_response_cuda.py \
  --junitxml="$output_root/pytest.xml" > "$output_root/pytest.log" 2>&1
