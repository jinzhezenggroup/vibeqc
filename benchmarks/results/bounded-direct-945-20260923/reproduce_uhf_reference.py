"""Produce a stable independent PySCF reference for the tetramer cation.

The symmetric atomic guess can converge to a higher UHF stationary point.
Internal stability rotations and a short Newton minimization select the lower
state without borrowing a VibeQC density. Ordinary SCF then refines its residual.
"""

import argparse
import json
from pathlib import Path

from pyscf import gto, scf

from benchmarks._cases import benchmark_cases

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
case = benchmark_cases()["water-tetramer-def2-tzvp-spherical"]
molecule = gto.M(
    atom=case.atoms,
    basis=case.pyscf_basis,
    unit="Bohr",
    cart=False,
    charge=1,
    spin=1,
    verbose=0,
)
reference = scf.UHF(molecule)
reference.conv_tol = 1e-12
reference.conv_tol_grad = 1e-7
reference.max_cycle = 200
reference.kernel()
assert reference.converged
trajectory = [float(reference.e_tot)]
for attempt in range(4):
    orbitals, _, stable, _ = reference.stability(
        internal=True, external=False, return_status=True
    )
    if stable:
        break
    minimize = reference.newton()
    minimize.conv_tol_grad = 1e-6
    minimize.max_cycle = 20
    minimize.kernel(dm0=reference.make_rdm1(orbitals, reference.mo_occ))
    density = minimize.make_rdm1()
    reference = scf.UHF(molecule)
    reference.conv_tol = 1e-12
    reference.conv_tol_grad = 1e-9
    reference.direct_scf_tol = 1e-16
    reference.max_cycle = 200
    reference.kernel(dm0=density)
    assert reference.converged
    trajectory.append(float(reference.e_tot))
else:
    raise RuntimeError("No internally stable UHF solution after four rotations")
forces = -reference.nuc_grad_method().kernel()
record = {
    "energy": float(reference.e_tot),
    "forces": forces.tolist(),
    "internal_stability": bool(stable),
    "trajectory": trajectory,
    "spin_square": reference.spin_square(),
    "origin": "Independent PySCF stability rotation, Newton minimization and SCF refinement; no VibeQC density used.",
}
args.output.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"energy": reference.e_tot, "internal_stability": bool(stable)}))
