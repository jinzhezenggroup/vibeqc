#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
root="$PWD/.artifacts/issues388-391"
out="$root/final/profiles"
mkdir -p "$out"
export VIBEQC_LIBRARY="$root/final/libvibeqc.so"
export LD_LIBRARY_PATH="$root/final:/group/software/cuda-12.9.1/lib64"

/tmp/vibeqc-pr-review-env/bin/python - <<'PYENV'
import importlib.metadata,json,os,platform,subprocess
from pathlib import Path
out=Path('.artifacts/issues388-391/final/profiles/environment.json')
commands={'gpu':['nvidia-smi','--query-gpu=name,driver_version,memory.total,pci.bus_id,power.limit','--format=csv,noheader'], 'nvcc':['/group/software/cuda-12.9.1/bin/nvcc','--version'], 'nsys':['/group/software/cuda-12.9.1/bin/nsys','--version'], 'host_cxx':['c++','--version']}
d={'host':platform.platform(),'python':platform.python_version(),'slurm_job_id':os.environ['SLURM_JOB_ID'],'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'tools':{k:subprocess.check_output(v,text=True).strip() for k,v in commands.items()},'packages':{}}
for name in ('numpy','pyscf','cupy-cuda12x','gpu4pyscf'):
 try:d['packages'][name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:d['packages'][name]=None
out.write_text(json.dumps(d,indent=2)+'\n')
PYENV
export VIBEQC_DF_RESIDENT_EXCHANGE=legacy
for aos in 768 384; do
  if [[ "$aos" == 384 ]]; then case_name=water-hexadecamer-2s4-def2-svp-spherical; else case_name=water-32mer-4s4-def2-svp-spherical; fi
  reference="benchmarks/results/issue377-379-df/gpu4pyscf/$case_name.json"
  nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits -lms 100 > "$out/$aos-memory.csv" &
  sampler_pid=$!
  trap 'kill "$sampler_pid" 2>/dev/null || true' EXIT
  /group/software/cuda-12.9.1/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=repeat:2 --export=sqlite --output "$out/$aos-diis" /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint --aos "$aos" --control VIBEQC_DF_DIIS_DOTS --policies serial auto --repeats 1 --host-trace --cuda-profile --reference "$reference" --cold-control VIBEQC_DF_RESIDENT_EXCHANGE=legacy --cold-control VIBEQC_DF_DIIS_DOTS=serial --warm-checkpoint-out "$out/$aos-diis.checkpoint" --output "$out/$aos-diis.json"
  unset VIBEQC_DF_RESIDENT_EXCHANGE
  /group/software/cuda-12.9.1/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite --output "$out/$aos-default" /tmp/vibeqc-pr-review-env/bin/python -m benchmarks.df_policy_endpoint --aos "$aos" --policies auto --repeats 1 --host-trace --cuda-profile --reference "$reference" --cold-control VIBEQC_DF_RESIDENT_EXCHANGE=legacy --cold-control VIBEQC_DF_DIIS_DOTS=serial --warm-checkpoint-out "$out/$aos-default.checkpoint" --output "$out/$aos-default.json"
  export VIBEQC_DF_RESIDENT_EXCHANGE=legacy
  kill "$sampler_pid"
  wait "$sampler_pid" || true
  trap - EXIT
done
# Small instrumented correctness checks stay outside clean endpoint samples.
unset VIBEQC_DF_RESIDENT_EXCHANGE
export VIBEQC_RESOURCE_CUDA_TEST=1
/group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck --error-exitcode 99 build/cuda-release-sm120/vibeqc_cuda_diis_tests
/group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck --error-exitcode 99 /tmp/vibeqc-pr-review-env/bin/python -m pytest -q -x tests/python/test_df_occupied_response_cuda.py -k 'packed_response_crosses or batched_full_response'
/group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck --error-exitcode 99 /tmp/vibeqc-pr-review-env/bin/python -m pytest -q -x tests/python/test_df_resident_response_cuda.py -k raw_view_binds
/tmp/vibeqc-pr-review-env/bin/python -m pytest -q -x tests/python/test_df_resident_response_cuda.py tests/python/test_df_occupied_response_cuda.py tests/python/test_df_shell_derivatives_cuda.py tests/python/test_df_response_weights_cuda.py tests/python/test_df_occupied_cuda.py tests/python/test_df_exchange_selector_cuda.py tests/python/test_cuda_density_fitting_source.py
set +e
/group/software/cuda-12.9.1/bin/ncu --set basic --launch-count 1 build/cuda-release-sm120/vibeqc_df_shell_pairs_tests > "$out/ncu.txt" 2>&1
printf '%s\n' "$?" > "$out/ncu-status.txt"
