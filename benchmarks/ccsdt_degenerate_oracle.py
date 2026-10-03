"""Pin exact tetrahedral methane with corrected PySCF triples Lambda."""

import argparse
import hashlib
import json
from pathlib import Path

import pyscf
from pyscf import cc, gto, scf
from pyscf.cc import ccsd_t_lambda
from pyscf.grad import ccsd_t

from benchmarks._retention import raw_output_path


def main() -> None:
    """Keep the independent corrected triples Lambda outside production."""
    assert pyscf.__version__ == "2.14.0"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    pack = (
        Path(__file__).resolve().parents[1] / "python/generativeqc/data/basis_pack.json"
    )
    elements = json.loads(pack.read_text())["bases"]["sto-3g"]["elements"]
    basis = {
        gto.mole._symbol(int(z)): [
            [
                s["angular_momentum"],
                *[
                    [float(e), float(c)]
                    for e, c in zip(s["exponents"], s["coefficients"])
                ],
            ]
            for s in elements[z]
        ]
        for z in ("1", "6")
    }
    atoms = [
        ("C", [0.0, 0.0, 0.0]),
        ("H", [1.2, 1.2, 1.2]),
        ("H", [1.2, -1.2, -1.2]),
        ("H", [-1.2, 1.2, -1.2]),
        ("H", [-1.2, -1.2, 1.2]),
    ]
    mol = gto.M(atom=atoms, basis=basis, unit="Bohr", cart=False, verbose=0)
    hf = scf.RHF(mol)
    hf.conv_tol = 1e-13
    hf.conv_tol_grad = 1e-11
    hf.max_cycle = 200
    hf.kernel()
    c = cc.CCSD(hf)
    c.conv_tol = 1e-13
    c.conv_tol_normt = 1e-11
    c.max_cycle = 150
    c.kernel()
    eris = c.ao2mo()
    triples = float(c.ccsd_t(eris=eris))
    converged, l1, l2 = ccsd_t_lambda.kernel(
        c, eris, c.t1, c.t2, max_cycle=150, tol=1e-11
    )
    assert hf.converged and c.converged and converged
    g = ccsd_t.Gradients(c)
    g.cphf_max_cycle = 100
    g.cphf_conv_tol = 1e-12
    force = -g.kernel(c.t1, c.t2, l1, l2, eris=eris)
    record = {
        "schema": "generativeqc.ccsdt-degenerate-oracle.v1",
        "pyscf": pyscf.__version__,
        "basis_pack_sha256": hashlib.sha256(pack.read_bytes()).hexdigest(),
        "units": {"coordinates": "bohr", "energy": "hartree", "forces": "hartree/bohr"},
        "inputs": atoms,
        "orbital_energies": hf.mo_energy.tolist(),
        "energy": float(c.e_tot + triples),
        "triples": triples,
        "forces": force.tolist(),
        "lambda": "pyscf.cc.ccsd_t_lambda.kernel corrected triples RHS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
