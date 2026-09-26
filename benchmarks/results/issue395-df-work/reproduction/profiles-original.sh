#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm}"
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
baseline=/home/jzzeng/codes/vibeqc-issues388-391/.artifacts/issues388-391/final
candidate="$PWD/.artifacts/issues392-395/work-v1"
out="$candidate/qualification"
export VIBEQC_LIBRARY="$candidate/libvibeqc.so"
export LD_LIBRARY_PATH="$candidate:/group/software/cuda-12.9.1/lib64"
export VIBEQC_DF_SHELL_COUNTERS=1
/tmp/vibeqc-pr-review-env/bin/python - <<'PY'
import importlib.metadata, json, os, platform, subprocess
from pathlib import Path
commands = {'gpu':['nvidia-smi','--query-gpu=name,driver_version,memory.total,pci.bus_id','--format=csv,noheader'],
            'nvcc':['/group/software/cuda-12.9.1/bin/nvcc','--version'],
            'nsys':['/group/software/cuda-12.9.1/bin/nsys','--version']}
result = {'slurm_job_id':os.environ['SLURM_JOB_ID'], 'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
          'host':platform.platform(), 'python':platform.python_version(),
          'tools':{name:subprocess.check_output(command,text=True).strip() for name,command in commands.items()},
          'packages':{name:importlib.metadata.version(name) for name in ('numpy','scipy','pyscf')}}
Path('.artifacts/issues392-395/work-v1/environment.json').write_text(json.dumps(result,indent=2)+'\n')
PY
reference=benchmarks/results/issue377-379-df/gpu4pyscf/water-32mer-4s4-def2-svp-spherical.json
checkpoint="$baseline/768-normal.checkpoint"
/tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint \
  --aos 768 --control VIBEQC_DF_SHELL_WORK --policies 0 1 --repeats 1 \
  --expected-iterations 3 --trace --reference "$reference" \
  --warm-checkpoint-in "$checkpoint" --skip-cold --output "$out/768-overhead.json"
/group/software/cuda-12.9.1/bin/nsys profile --trace=cuda,nvtx --sample=none \
  --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi \
  --capture-range-end=repeat:2 --export=sqlite --output "$out/768-work" \
  /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint \
  --aos 768 --control VIBEQC_DF_SHELL_WORK --policies 0 1 --repeats 1 \
  --expected-iterations 3 --trace --cuda-profile --reference "$reference" \
  --warm-checkpoint-in "$checkpoint" --skip-cold --output "$out/768-profile.json"
/tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_shell_work_ledger \
  --trace "$out/768-profile.0-1.jsonl" --measurement "$out/768-profile.json" \
  --generated-header "$candidate/generated_df_shell_derivatives.cuh" \
  --nsys "$out/768-work.2.sqlite" --output "$out/768-work-ledger.json"
