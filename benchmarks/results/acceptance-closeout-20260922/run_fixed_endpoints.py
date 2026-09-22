"""Replay the complete retained HF/DF/MP2 suite after test-only contract repairs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/jzzeng/codes/vibeqc-acceptance-fixes-20260922')
BASE = Path('/home/jzzeng/codes/vibeqc-acceptance-closeout-20260922')
OUT = Path(__file__).resolve().parent
assert os.environ.get('SLURM_JOB_ID')
historical = json.loads((BASE/'benchmarks/results/scf-direct-native-rtx5090/integration.json').read_text())
env = os.environ.copy()
env.update({k: v for k, v in historical['environment'].items() if k not in ('PYTHONPATH','VIBEQC_LIBRARY')})
env.update(PYTHONPATH=f'{ROOT}/python:{ROOT}', VIBEQC_LIBRARY=str(BASE/'build-acceptance/libvibeqc.so'),
           LD_LIBRARY_PATH='/group/software/cuda-12.9.1/lib64:'+env.get('LD_LIBRARY_PATH',''))
tests = [x for x in historical['validation'][0]['commands'][1]['command'] if x.startswith('tests/python/')]
tests += ['tests/python/test_direct_force_state_cuda.py']
command = [sys.executable, '-m', 'pytest', '-q', '-rs', *tests, f'--junitxml={OUT}/python-hf-df-mp2-fixed.xml']
start = time.monotonic()
with (OUT/'python-hf-df-mp2-fixed.log').open('w') as log:
    code = subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=850).returncode
(OUT/'fixed-endpoints.json').write_text(json.dumps({'command':command,'returncode':code,'seconds':time.monotonic()-start,
    'slurm_job':os.environ['SLURM_JOB_ID'],'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
    'production_source':'0d89ab6fb6219d9af6641d5cfafe0b43f8387206',
    'test_delta':'Explicit energy request in test_mp2_extended.py; no production change'},indent=2)+'\n')
print('fixed endpoints',code,flush=True)
