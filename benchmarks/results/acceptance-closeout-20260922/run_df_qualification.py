"""Run the existing #439/#949 endpoint protocol without relaxing its gates.

The panel controls isolate scalar versus BLAS charge reductions on the same
response storage contract. A second, unforced run measures current automatic
production selection. Intrusive counters are collected only after clean timing.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--aos', choices=('384', '768'), required=True)
args = parser.parse_args()
assert os.environ.get('SLURM_JOB_ID')
root = Path('/home/jzzeng/codes/vibeqc-acceptance-closeout-20260922')
out = Path(__file__).resolve().parent
env = os.environ.copy()
env.update(PYTHONPATH=f'{root}/python:{root}',
           VIBEQC_LIBRARY=str(root/'build-acceptance/libvibeqc.so'),
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           LD_LIBRARY_PATH='/group/software/cuda-12.9.1/lib64:'+env.get('LD_LIBRARY_PATH',''))
report = []
for mode in ('panel-ablation', 'automatic'):
    selected = env.copy()
    command = [sys.executable, 'benchmarks/df_policy_endpoint.py', '--aos', args.aos,
               '--repeats', '7', '--cpu-reference', '--components-after',
               '--output', str(out/f'df-{args.aos}-{mode}.json')]
    if mode == 'panel-ablation':
        selected['VIBEQC_DF_RESPONSE_STORAGE'] = 'panel'
        command += ['--control', 'VIBEQC_DF_SERIAL_RESPONSE_DOT', '--policies', '0', '1']
    else:
        command += ['--control', 'VIBEQC_DF_RESPONSE_SPACE', '--policies', 'auto']
    start = time.monotonic()
    with (out/f'df-{args.aos}-{mode}.log').open('w') as log:
        try:
            code = subprocess.run(command, cwd=root, env=selected, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=1200).returncode
        except subprocess.TimeoutExpired:
            code = 124
    report.append({'mode': mode, 'command': command, 'returncode': code,
                   'seconds': time.monotonic()-start, 'slurm_job': os.environ['SLURM_JOB_ID']})
    (out/f'df-{args.aos}-runs.json').write_text(json.dumps(report,indent=2)+'\n')
    print(args.aos, mode, code, report[-1]['seconds'], flush=True)
