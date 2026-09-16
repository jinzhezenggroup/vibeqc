#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset VIBEQC_DF_SHELL_WORK VIBEQC_DF_SHELL_COUNTERS VIBEQC_DF_TRACE VIBEQC_DF_HOST_TRACE VIBEQC_DF_PROGRESS_TRACE
export VIBEQC_LIBRARY="$PWD/.artifacts/issue394-000/v1/libvibeqc.so"
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64
mkdir -p .artifacts/issue394-000/profiles
/tmp/vibeqc-pr-review-env/bin/python - <<'PY'
import importlib.metadata,json,os,platform,subprocess
from pathlib import Path
commands={'gpu':['nvidia-smi','--query-gpu=name,driver_version,memory.total,pci.bus_id,power.limit','--format=csv,noheader'], 'nvcc':['/group/software/cuda-12.9.1/bin/nvcc','--version'], 'nsys':['/group/software/cuda-12.9.1/bin/nsys','--version'], 'host_cxx':['c++','--version']}
data={'host':platform.platform(),'python':platform.python_version(),'slurm_job_id':os.environ['SLURM_JOB_ID'],'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'tools':{k:subprocess.check_output(v,text=True).strip() for k,v in commands.items()},'packages':{}}
for name in ('numpy','pyscf','cupy-cuda12x','gpu4pyscf'):
 try:data['packages'][name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:data['packages'][name]=None
Path('.artifacts/issue394-000/profiles/environment.json').write_text(json.dumps(data,indent=2)+'\n')
PY
nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits -lms 100 > .artifacts/issue394-000/profiles/memory.csv &
sampler_pid=$!
trap 'kill "$sampler_pid" 2>/dev/null || true' EXIT
/usr/bin/time -v -o .artifacts/issue394-000/profiles/process-residency.txt /group/software/cuda-12.9.1/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=repeat:4 --export=sqlite --output .artifacts/issue394-000/profiles/000 /tmp/vibeqc-pr-review-env/bin/python .artifacts/issue394-000/profile.py
kill "$sampler_pid"
wait "$sampler_pid" || true
trap - EXIT
