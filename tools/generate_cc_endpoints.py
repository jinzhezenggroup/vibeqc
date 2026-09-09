"""Pinned PySCF/FCI endpoint generation using the exact #138 basis inputs."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from tools.cc_endpoint_fixtures import array_hash, cases
from tools.generate_validation_references import pyscf_molecule
from tools.vibeqc_validation.schema import canonical_hash, file_hash


def generate(output, compare=None):
    import pyscf
    from pyscf import ao2mo, cc, fci, scf
    from pyscf.cc import ccsd, rccsd
    from threadpoolctl import threadpool_info, threadpool_limits

    if pyscf.__version__ != "2.14.0":
        raise ValueError("CC endpoints require PySCF 2.14.0")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for inputs in cases():
        with threadpool_limits(limits=1):
            mol, scale, _ = pyscf_molecule(inputs)
            mf = scf.RHF(mol)
            mf.conv_tol = 1e-13
            mf.conv_tol_grad = 1e-11
            mf.max_cycle = 150
            mf.kernel()
            if not mf.converged:
                raise ValueError("reference RHF did not converge")
            for col in range(len(mf.mo_energy)):
                pivot = np.argmax(np.abs(mf.mo_coeff[:, col] / scale))
                if mf.mo_coeff[pivot, col] < 0:
                    mf.mo_coeff[:, col] *= -1
            coupled = cc.CCSD(mf)
            coupled.conv_tol = 1e-13
            coupled.conv_tol_normt = 1e-11
            coupled.max_cycle = 150
            coupled.kernel()
            if not coupled.converged:
                raise ValueError("reference CCSD did not converge")
            C = mf.mo_coeff / scale[:, None]
            S = mf.get_ovlp() * scale[:, None] * scale[None, :]
            F = mf.get_fock() * scale[:, None] * scale[None, :]
            h = mf.get_hcore() * scale[:, None] * scale[None, :]
            g = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), len(mf.mo_energy))
            ao = mol.intor("int2e") * np.einsum(
                "p,q,r,s->pqrs", scale, scale, scale, scale
            )
            arrays = {
                "S": S,
                "h": h,
                "F": F,
                "C": C,
                "eps": mf.mo_energy,
                "occ": mf.mo_occ,
                "g": g,
                "ao": ao,
                "t1": coupled.t1,
                "t2": coupled.t2,
            }
            eris = coupled.ao2mo()
            update = coupled.update_amps(coupled.t1, coupled.t2, eris)
            o = coupled.nocc
            d1 = mf.mo_energy[:o, None] - mf.mo_energy[None, o:]
            d2 = d1[:, None, :, None] + d1[None, :, None, :]
            residual = max(
                np.max(np.abs(d1 * (update[0] - coupled.t1))),
                np.max(np.abs(d2 * (update[1] - coupled.t2))),
            )
            if residual > 1e-9:
                raise ValueError("independent CC reference physical residual too large")
            record = {
                "version": 1,
                "inputs": inputs,
                "inputs_hash": canonical_hash(inputs),
                "arrays_hash": array_hash(arrays),
                "pyscf": pyscf.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "hf_energy": float(mf.e_tot),
                "correlation_energy": float(coupled.e_corr),
                "total_energy": float(coupled.e_tot),
                "scf_residual": float(
                    np.max(np.abs(C.T @ F @ C - np.diag(mf.mo_energy)))
                ),
                "physical_cc_residual": float(residual),
                "nuclear_repulsion": float(mol.energy_nuc()),
                "generator_sha256": file_hash(__file__),
                "threads": threadpool_info(),
                "source_hashes": {
                    "ccsd.py": file_hash(ccsd.__file__),
                    "rccsd.py": file_hash(rccsd.__file__),
                    "fci_direct_spin1.py": file_hash(fci.direct_spin1.__file__),
                },
            }
            if mol.nelectron == 2:
                efci, _ = fci.direct_spin1.kernel(
                    mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff,
                    g,
                    len(mf.mo_energy),
                    mol.nelec,
                    ecore=mol.energy_nuc(),
                    tol=1e-13,
                )
                record["fci_total_energy"] = float(efci)
                if abs(efci - coupled.e_tot) > 1e-8:
                    raise ValueError("two-electron CCSD/FCI disagreement")
            if compare:
                previous = json.loads(
                    (Path(compare) / (inputs["name"] + ".json")).read_text()
                )
                if (
                    previous["arrays_hash"] != record["arrays_hash"]
                    or abs(previous["total_energy"] - record["total_energy"]) > 1e-12
                ):
                    raise ValueError("endpoint two-generation stability failed")
                record["stability"] = {
                    "arrays_identical": True,
                    "energy_error": abs(
                        previous["total_energy"] - record["total_energy"]
                    ),
                }
            np.savez_compressed(output / (inputs["name"] + ".npz"), **arrays)
            (output / (inputs["name"] + ".json")).write_text(
                json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            print(
                inputs["name"],
                record["total_energy"],
                record["physical_cc_residual"],
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    generate(args.output, args.compare)
