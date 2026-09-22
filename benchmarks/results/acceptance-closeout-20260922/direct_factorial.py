"""Matched #669 controls with a frozen density and a separate GPU4PySCF oracle.

Forced final-Fock reconstruction is an explicit current fallback control, not a
reconstruction of the historical pre-#698 default. The two controls independently
toggle derivative scheduling and force-state reuse; every timed call requests
energy and forces. GPU4PySCF starts only after clean native timing has finished.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
from benchmarks._cases import benchmark_cases
from vibeqc import Calculator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--aos', type=int, choices=(384, 768), required=True)
parser.add_argument('--profile', choices=('historical', 'strict'), required=True)
parser.add_argument('--process', type=int, required=True)
parser.add_argument('--repeats', type=int, default=7)
args = parser.parse_args()
assert os.environ.get('SLURM_JOB_ID'), 'GPU execution requires Slurm'
root = Path('/home/jzzeng/codes/vibeqc-acceptance-closeout-20260922')
out = Path(__file__).resolve().parent / f'direct-{args.aos}-{args.profile}-{args.process}.json'
case = benchmark_cases()[{384: 'water-hexadecamer-2s4-def2-svp-spherical', 768: 'water-32mer-4s4-def2-svp-spherical'}[args.aos]]
controls = dict(zip(('energy_tolerance', 'density_tolerance', 'screening_tolerance'),
                    (1e-10, 1e-9, 1e-12) if args.profile == 'historical' else (1e-12, 1e-10, 1e-14)))
arms = {
    'shell_warp+rebuild': ('shell_warp', '1'),
    'nucleus+rebuild': ('nucleus_cooperative', '1'),
    'shell_warp+reuse': ('shell_warp', '0'),
    'nucleus+reuse': ('nucleus_cooperative', '0'),
}
runtime = ctypes.CDLL('libcudart.so.12')
runtime.cudaDeviceSynchronize.restype = ctypes.c_int


def select(arm):
    """Select only the two diagnostic axes; no convergence policy is modified."""
    mapping, rebuild = arms[arm]
    os.environ['VIBEQC_ONE_ELECTRON_DERIVATIVES'] = 'generated'
    os.environ['VIBEQC_ONE_ELECTRON_DERIVATIVE_MAPPING'] = mapping
    os.environ['VIBEQC_FINAL_FOCK_REBUILD'] = rebuild


def sync():
    assert runtime.cudaDeviceSynchronize() == 0


report = {
    'source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'library_sha256': hashlib.file_digest(Path(os.environ['VIBEQC_LIBRARY']).open('rb'), 'sha256').hexdigest(),
    'slurm_job': os.environ['SLURM_JOB_ID'], 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'aos': args.aos, 'profile': args.profile, 'process': args.process,
    'controls': controls, 'atoms': case.atoms, 'basis': 'def2-svp', 'basis_representation': 'spherical',
    'warm_policy': 'fixed post-cold engine-local density; force properties throughout; prime every control transition',
    'historical_control_warning': 'Forced reconstruction is a current fallback, not the pre-#698 legacy default.',
    'predeclared_gates': {'energy_error': 1e-9, 'force_error': 1e-8, 'baseline_repeat_relative_spread': 0.05,
                         'timing_conclusion': 'No speed claim unless baseline controls are stable, branches match, and paired bootstrap CI excludes one.'},
    'protocol_version': 2,
    'samples': [],
    'priming': [],
}


def save():
    out.write_text(json.dumps(report, indent=2, default=str)+'\n')


def scientific_record(item):
    """Retain cold and priming outputs under the same oracle gate as warm calls."""
    return dict(energy=item.energy, forces=item.forces.tolist(),
                iterations=item.iterations, physical_residual_rms=item.physical_residual_rms,
                status=item.status, converged=item.converged, fock_builds=item.fock_builds, precision=item.precision)


report['hardware_before'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,clocks.sm,clocks.mem,power.draw,power.limit', '--format=csv'], text=True)
save()
select('nucleus+reuse')
calc = Calculator(method='rhf', basis=case.vibeqc_basis, basis_representation='spherical',
                  density_fitting='none', device='cuda', max_iterations=200, **controls)
with calc.prepare_batch([case.atoms], warm_start=True) as batch:
    cold_start = time.perf_counter()
    cold = batch.execute(strict=True, properties=('energy', 'forces'))
    sync()
    report['cold_seconds'] = time.perf_counter()-cold_start
    report['cold_iterations'] = cold.items[0].iterations
    report['cold'] = scientific_record(cold.items[0])
    save()
    batch.set_warm_start_updates(False)
    for arm in arms:
        select(arm)
        prime = batch.execute(strict=True, properties=('energy', 'forces'))
        sync()
        report['priming'].append(dict(arm=arm, **scientific_record(prime.items[0])))
        save()
    schedule = [('noise', 'nucleus+reuse')] * 8
    for repeat in range(args.repeats):
        order = list(arms)
        if (repeat + args.process) % 2:
            order.reverse()
        schedule += [(str(repeat), arm) for arm in order]
    previous_arm = 'nucleus+reuse'
    for repeat, arm in schedule:
        select(arm)
        sync()
        if arm != previous_arm:
            # Control changes can invalidate reusable operator state even when
            # the seed is frozen. Retain that transition cost separately.
            prime_start = time.perf_counter()
            prime = batch.execute(strict=True, properties=('energy', 'forces'))
            sync()
            report['priming'].append(dict(arm=arm, repeat=repeat,
                seconds=time.perf_counter()-prime_start, **scientific_record(prime.items[0])))
            save()
        previous_arm = arm
        started = time.perf_counter()
        result = batch.execute(strict=True, properties=('energy', 'forces'))
        sync()
        elapsed = time.perf_counter()-started
        item = result.items[0]
        report['samples'].append(dict(repeat=repeat, arm=arm, seconds=elapsed,
            iterations=item.iterations, energy_change=item.energy_change, density_rms=item.density_rms,
            physical_residual_rms=item.physical_residual_rms, energy=item.energy, forces=item.forces.tolist(),
            converged=item.converged, status=item.status, fock_builds=item.fock_builds, precision=item.precision,
            eigensolvers=[str(x) for x in batch.last_eigensolver_diagnostics()]))
        save()
        print(args.aos, args.profile, repeat, arm, f'{elapsed:.6f}', item.iterations, flush=True)

# Reference construction and its allocator are outside every clean timing arm.
from pyscf import gto
from gpu4pyscf.scf import RHF
mol = gto.M(atom=case.atoms, basis='def2-svp', unit='Bohr', cart=False, verbose=0)
assert mol.nao == args.aos
mf = RHF(mol)
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-10
mf.direct_scf_tol = 1e-14
mf.max_cycle = 200
energy = float(mf.kernel())
assert mf.converged
gradient = mf.nuc_grad_method().kernel()
if hasattr(gradient, 'get'):
    gradient = gradient.get()
report['independent_reference'] = {'implementation': 'GPU4PySCF RHF / def2-SVP / spherical', 'energy': energy, 'forces': (-gradient).tolist()}
observations = [report['cold'], *report['priming'], *report['samples']]
assert all(np.asarray(r['forces']).shape == gradient.shape and np.isfinite(r['energy'])
           and np.isfinite(r['forces']).all() for r in observations)
report['maximum_energy_error'] = max(abs(r['energy']-energy) for r in observations)
report['maximum_force_error'] = max(float(np.max(np.abs(np.asarray(r['forces'])+gradient))) for r in observations)
noise = [r['seconds'] for r in report['samples'] if r['repeat'] == 'noise']
report['noise_relative_spread'] = (max(noise)-min(noise))/statistics.median(noise)
report['medians'] = {arm: statistics.median(r['seconds'] for r in report['samples'] if r['repeat'] != 'noise' and r['arm'] == arm) for arm in arms}
report['hardware_after'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,clocks.sm,clocks.mem,power.draw,power.limit', '--format=csv'], text=True)
report['numerical_pass'] = report['maximum_energy_error'] <= 1e-9 and report['maximum_force_error'] <= 1e-8
report['completed'] = True
save()
print(json.dumps({k: report[k] for k in ('medians','maximum_energy_error','maximum_force_error','numerical_pass')}), flush=True)
