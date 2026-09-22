"""Replay the retained #240 acceptance commands on one immutable build.

Each command has its own log and finite timeout. Failures are retained while
independent suites continue, so one unrelated native failure cannot erase the
HF/DF/MP2 acceptance evidence.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/jzzeng/codes/vibeqc-acceptance-closeout-20260922')
OUT = Path(__file__).resolve().parent
BUILD = ROOT / 'build-acceptance'
assert os.environ.get('SLURM_JOB_ID'), 'GPU execution requires Slurm'
historical = json.loads((ROOT / 'benchmarks/results/scf-direct-native-rtx5090/integration.json').read_text())
env = os.environ.copy()
env.update({k: v for k, v in historical['environment'].items()
            if k not in ('PYTHONPATH', 'VIBEQC_LIBRARY')})
env.update(PYTHONPATH=f'{ROOT}/python:{ROOT}', VIBEQC_LIBRARY=str(BUILD / 'libvibeqc.so'),
           CUDACXX='/group/software/cuda-12.9.1/bin/nvcc', VIBEQC_DFT_CUDA_TEST='1',
           LD_LIBRARY_PATH='/group/software/cuda-12.9.1/lib64:' + env.get('LD_LIBRARY_PATH', ''))
original = historical['validation'][0]['commands'][1]['command']
tests = [x for x in original if x.startswith('tests/python/')]
tests += ['tests/python/test_direct_force_state_cuda.py']
commands = [
    ('native', ['ctest', '--test-dir', str(BUILD), '--output-on-failure', '--timeout', '300', '-j', '1'], 2400),
    ('python-hf-df-mp2', [sys.executable, '-m', 'pytest', '-q', '-rs', *tests, f'--junitxml={OUT}/python-hf-df-mp2.xml'], 1800),
    ('weighted', [sys.executable, 'tools/validate_weighted_eri.py', '--probe', str(BUILD / 'vibeqc_weighted_eri_probe'), '--output', str(OUT / 'weighted')], 1200),
    ('r2scan3c', [sys.executable, '-m', 'pytest', '-q', '-rs', 'tests/python/test_r2scan3c_execution.py', 'tests/python/test_r2scan3c_public_api.py', 'tests/python/test_r2scan3c_cpu_composition.py', 'tests/python/test_r2scan3c_method.py', f'--junitxml={OUT}/r2scan3c.xml'], 1200),
    ('public-dft', [sys.executable, '-m', 'pytest', '-q', '-rs',
      'tests/python/test_dft_complete_cuda.py::test_public_cuda_calculator_forces_match_independent_gradient',
      'tests/python/test_dft_complete_cuda.py::test_public_cuda_prepared_force_replay_retains_execution',
      'tests/python/test_dft_complete_cuda.py::test_public_cuda_batch_changed_geometry_and_failure_isolation',
      f'--junitxml={OUT}/public-dft.xml'], 1200),
]
report = {
    'source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'library_sha256': hashlib.file_digest((BUILD / 'libvibeqc.so').open('rb'), 'sha256').hexdigest(),
    'slurm_job': os.environ['SLURM_JOB_ID'],
    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'hardware': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,clocks.sm,clocks.mem,power.limit', '--format=csv'], text=True),
    'commands': [],
}
for name, command, timeout in commands:
    start = time.monotonic()
    with (OUT / f'{name}.log').open('w') as log:
        try:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = 124
    report['commands'].append({'name': name, 'command': command, 'returncode': code, 'seconds': time.monotonic()-start})
    (OUT / 'runtime.json').write_text(json.dumps(report, indent=2)+'\n')
    print(name, code, report['commands'][-1]['seconds'], flush=True)
