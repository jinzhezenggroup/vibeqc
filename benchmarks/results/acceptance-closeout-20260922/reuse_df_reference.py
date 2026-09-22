"""Reuse the completed strict independent oracle without changing native timing.

The original 768-AO scalar campaign hit its time limit after the CPU oracle and
one BLAS sample. This adapter validates that retained reference's full geometry,
bases, charge model and numerical settings, then delegates all native execution,
priming, error checks and counters to the unchanged pinned endpoint runner.
It does not manufacture a GPU4PySCF record or relax the original oracle gates.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path('/home/jzzeng/codes/vibeqc-acceptance-closeout-20260922')
OUT = Path(__file__).resolve().parent
assert os.environ.get('SLURM_JOB_ID')
sys.path[:0] = [str(ROOT), str(ROOT/'python')]
from benchmarks import df_policy_endpoint as endpoint

reference_path = OUT/'df-768-panel-ablation.json'
reference = json.loads(reference_path.read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == reference['git_head']
assert hashlib.sha256(Path(os.environ['VIBEQC_LIBRARY']).read_bytes()).hexdigest() == reference['library_sha256']


def cached_reference(case, orbital_basis, auxiliary_basis):
    """Return exactly the completed CPU oracle after binding all scientific inputs."""
    retained = reference['cpu_reference']
    settings = reference['scientific_settings']
    assert json.loads(json.dumps(case.atoms)) == settings['geometries_bohr']
    assert case.method == settings['method'] == 'rhf'
    assert case.charge == 0 and case.multiplicity == 1
    assert case.basis_representation == settings['basis_representation'] == 'spherical'
    assert orbital_basis == retained['orbital_basis'] == settings['basis']
    assert auxiliary_basis == retained['auxiliary_basis'] == settings['auxiliary_basis']
    assert retained['ao_count'] == retained['auxiliary_count'] == 768
    assert retained['energy_tolerance'] == 1e-13
    assert retained['gradient_tolerance'] == 1e-12
    assert retained['direct_scf_tolerance'] == 1e-14
    assert retained['auxbasis_response'] is True
    energy = np.asarray(retained['energies_hartree'])
    force = np.asarray(retained['forces_hartree_per_bohr'])
    assert energy.shape == (1,) and force.shape == (1,len(case.atoms),3)
    assert np.isfinite(energy).all() and np.isfinite(force).all()
    return energy, force, retained


endpoint.cpu_reference = cached_reference
output = Path(sys.argv[sys.argv.index('--output')+1])
try:
    endpoint.main()
finally:
    if output.exists():
        payload = json.loads(output.read_text())
        payload['reference_reuse'] = {
            'receipt': reference_path.name,
            'sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'note': 'Exact completed independent CPU reference reused after strict scientific identity checks; no new CPU solve in this run.',
        }
        output.write_text(json.dumps(payload,indent=2)+'\n')
