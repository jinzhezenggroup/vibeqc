#!/usr/bin/env bash
set -euo pipefail
cd /home/jzzeng/codes/vibeqc-issue-206-response-metric-gemv
[[ -n "$SLURM_JOB_ID" && -n "$CUDA_VISIBLE_DEVICES" ]]
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VIBEQC_LIBRARY=$PWD/.artifacts/issue206-metric-gemv/frozen-attempt1/libvibeqc.so
export LD_LIBRARY_PATH=$PWD/.artifacts/issue206-metric-gemv/frozen-attempt1:/group/software/cuda-12.9.1/lib64:/tmp/vibeqc-308-gpu4pyscf-env/lib/python3.13/site-packages/cutensor/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
printf 'Slurm job: %s\n' "$SLURM_JOB_ID"
sha256sum "$VIBEQC_LIBRARY"
nvidia-smi --query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,power.limit --format=csv
direct_status=0
/tmp/vibeqc-308-gpu4pyscf-env/bin/python benchmarks/real_molecule_gate.py --density-fitting none --repeats 5 --output-directory .artifacts/issue206-metric-gemv/direct-gate || direct_status=$?
exit "$direct_status"
