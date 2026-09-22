"""Qualify bounded canonical r2SCAN-3c totals, replay and resource evidence.

Independent totals combine PySCF/Libxc electronic gradients on the identical
grid, DFT-D4 4.2.0 with the defining r2SCAN-3c charge model, and the retained
independent gCP reference. Failed cells and the s/p force-domain limit are kept.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/jzzeng/codes/vibeqc-acceptance-closeout-20260922')
OUT = Path(__file__).resolve().parent
sys.path.extend([str(OUT/'oracle-py311'), str(ROOT/'tests/python')])
assert os.environ.get('SLURM_JOB_ID')

import numpy as np
import dftd4
from dftd4.interface import DispersionModel, DampingParam
from test_dft_complete_cpu import independent_semilocal_total_gradient
from tools.vibeqc_gcp.reference import evaluate_r2scan3c_gcp
from vibeqc import Calculator, KsOptions, load_r2scan3c_basis
from vibeqc._dft_gradient import StationaryKsState
from vibeqc_compiler.dft import NativeAO

H2 = [('H', (0.1,-0.1,-0.7)), ('H', (0.0,0.1,0.8))]
H3 = H2 + [('H', (1.6,0.2,0.0))]
H4 = H2 + [('H', (4.5,0.2,-0.8)), ('H', (4.4,0.3,0.7))]
CASES = [('h2', H2, 0, 1), ('h3-cation', H3, 1, 1),
         ('h2-dimer', H4, 0, 1), ('h3-doublet', H3, 0, 2)]
PARAM = DampingParam(s6=1., s8=0., s9=2., a1=.42, a2=5.65)
BASIS = load_r2scan3c_basis()
report = {'source_sha': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
          'library_sha256': hashlib.file_digest(Path(os.environ['VIBEQC_LIBRARY']).open('rb'),'sha256').hexdigest(),
          'slurm_job': os.environ['SLURM_JOB_ID'], 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
          'dftd4_version': dftd4.__version__, 'basis_identity': BASIS.identity,
          'grid': 'production default KsOptions',
          'gates': {'energy': 2e-9, 'force': 1e-7, 'finite_difference': 1e-6}, 'cases': []}


def save():
    (OUT/'r2scan3c-qualification.json').write_text(json.dumps(report,indent=2,default=str)+'\n')


def calculator(multiplicity):
    return Calculator(method='r2scan-3c-rks' if multiplicity == 1 else 'r2scan-3c-uks',
                      device='cuda', ks_options=KsOptions(), max_iterations=200,
                      energy_tolerance=1e-12, density_tolerance=1e-10)


def record(result, elapsed, batch):
    item = result.items[0]
    return {'seconds': elapsed, 'energy': item.energy, 'forces': item.forces.tolist(),
            'iterations': item.iterations, 'physical_residual_rms': item.physical_residual_rms,
            'correction_backend': item.dispersion.backend,
            'd4_energy': item.dispersion.d4.energy, 'gcp_energy': item.dispersion.gcp.energy,
            'resources': copy.deepcopy(batch.resource_diagnostics)}


for name, atoms, charge, multiplicity in CASES:
    row = {'case': name, 'atoms': atoms, 'charge': charge, 'multiplicity': multiplicity, 'phases': []}
    report['cases'].append(row)
    try:
        calc = calculator(multiplicity)
        with calc.prepare_batch([atoms], charges=[charge], multiplicities=[multiplicity]) as batch:
            for phase in ('cold','warm-1','warm-2','warm-3'):
                start=time.perf_counter()
                value=batch.execute(strict=True, properties=('energy','forces'))
                row['phases'].append({'phase':phase, **record(value,time.perf_counter()-start,batch)})
                if phase == 'cold': batch.set_warm_start_updates(False)
                save()
            with NativeAO(atoms,basis=BASIS,representation='spherical',charge=charge,multiplicity=multiplicity) as basis:
                row['aos']=basis.nao
                state=StationaryKsState.from_native(batch,basis)
                try:
                    ref_e, ref_g=independent_semilocal_total_gradient(basis,state,'r2scan-rks' if multiplicity==1 else 'r2scan-uks')
                finally:
                    state._source.close()
            xyz=np.asarray([x[1] for x in atoms])
            numbers=np.ones(len(atoms),dtype=int)
            d4=DispersionModel(numbers,xyz,charge=float(charge),ga=2.,gc=1.).get_dispersion(PARAM,grad=True)
            gcp=evaluate_r2scan3c_gcp(tuple(numbers),xyz)
            expected_e=ref_e+float(d4['energy'])+gcp.energy
            expected_f=-ref_g-d4['gradient']-gcp.gradient
            row['reference']={'energy':expected_e,'forces':expected_f.tolist(),
                              'd4_energy':float(d4['energy']),'gcp_energy':gcp.energy}
            row['maximum_energy_error']=max(abs(x['energy']-expected_e) for x in row['phases'])
            row['maximum_force_error']=max(float(np.max(np.abs(np.asarray(x['forces'])-expected_f))) for x in row['phases'])
            assert row['maximum_energy_error'] < 2e-9
            assert row['maximum_force_error'] < 1e-7
            moved=xyz.copy(); moved[-1]+=[.002,-.001,.003]
            start=time.perf_counter()
            changed=batch.execute(coordinates=[moved],strict=True,properties=('energy','forces'))
            row['phases'].append({'phase':'changed',**record(changed,time.perf_counter()-start,batch)})
        fresh_atoms=[(a[0],tuple(x)) for a,x in zip(atoms,moved)]
        fresh=calc.singlepoint(fresh_atoms,charge=charge,multiplicity=multiplicity,properties=('energy','forces'))
        row['changed_fresh_energy_error']=abs(changed.items[0].energy-fresh.energy)
        row['changed_fresh_force_error']=float(np.max(np.abs(changed.items[0].forces-fresh.forces)))
        assert row['changed_fresh_energy_error'] < 2e-9
        assert row['changed_fresh_force_error'] < 1e-7
        if name in ('h2','h3-cation'):
            direction=np.arange(xyz.size,dtype=float).reshape(xyz.shape)/xyz.size-.4
            analytic=-float(np.sum(np.asarray(row['phases'][0]['forces'])*direction))
            row['finite_differences']=[]
            for step in (1e-3,3e-4,1e-4):
                energies=[]
                for sign in (1,-1):
                    displaced=[(a[0],tuple(x)) for a,x in zip(atoms,xyz+sign*step*direction)]
                    energies.append(calc.singlepoint(displaced,charge=charge,multiplicity=multiplicity,properties=('energy',)).energy)
                fd=(energies[0]-energies[1])/(2*step)
                row['finite_differences'].append({'step':step,'analytic':analytic,'finite_difference':fd,'error':abs(fd-analytic)})
                save()
            assert row['finite_differences'][-1]['error'] < 1e-6
        row['passed']=True
    except Exception as error:
        row.update(passed=False,error=repr(error),traceback=traceback.format_exc())
    save()
    print(name,row['passed'],row.get('error',''),flush=True)

try:
    calc=calculator(1)
    with calc.prepare_batch([H2,H3],charges=[0,1],multiplicities=[1,1]) as batch:
        first=batch.execute(strict=True,properties=('energy','forces'))
        isolated=batch.execute(coordinates=[[0.],None],properties=('energy','forces'))
        assert not isolated.items[0].succeeded and isolated.items[0].forces is None
        assert isolated.items[1].succeeded and np.isfinite(isolated.items[1].forces).all()
        report['ragged_failure_isolation']={'passed':True,'statuses':[x.status for x in isolated.items]}
except Exception as error:
    report['ragged_failure_isolation']={'passed':False,'error':repr(error),'traceback':traceback.format_exc()}
save()

water=[('O',(0.,0.,0.)),('H',(1.4,0.,1.1)),('H',(-1.3,.1,1.2))]
try:
    calculator(1).singlepoint(water,properties=('energy','forces'))
    report['higher_angular_boundary']={'water_force':'admitted'}
except Exception as error:
    report['higher_angular_boundary']={'water_force':'rejected','error':repr(error)}
save()
