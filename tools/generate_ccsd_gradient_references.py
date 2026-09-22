"""Regenerate tiny independent PySCF 2.14.0 CCSD energy/gradient oracles.

External PySCF is used only by this explicit generator, never by the native
CCSD gradient consumer. Tighten the existing CPHF solver's tolerance, without
changing its equations, to avoid treating default response error as truth.
"""

import argparse
import hashlib
import inspect
import json
import typing
from pathlib import Path
from unittest.mock import patch

import numpy as np
from vibeqc_compiler.common.evidence import canonical_hash

from tools.cc_gradient_fixtures import CASES, ROOT, inputs
from tools.generate_validation_references import pyscf_molecule


def generate(name: typing.Any) -> typing.Any:
    import pyscf
    from pyscf import cc, scf
    from pyscf.grad import ccsd as grad_ccsd
    from pyscf.scf import cphf

    if pyscf.__version__ != "2.14.0":
        raise RuntimeError("gradient references require exactly PySCF 2.14.0")
    value = inputs(name)
    mol, _, _ = pyscf_molecule(value)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 150
    mf.direct_scf_tol = 1e-14
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("independent HF failed")
    coupled = cc.CCSD(mf, frozen=0)
    coupled.conv_tol, coupled.conv_tol_normt, coupled.max_cycle = 1e-13, 1e-11, 150
    coupled.kernel()
    if not coupled.converged:
        raise RuntimeError("independent CCSD failed")
    coupled.solve_lambda()
    if not coupled.converged_lambda:
        raise RuntimeError("independent Lambda failed")
    default_gradient = coupled.nuc_grad_method().kernel()
    original = cphf.solve

    def tight(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return original(*args, **{**kwargs, "tol": 1e-13, "max_cycle": 100})

    with patch.object(cphf, "solve", tight):
        gradient = coupled.nuc_grad_method().kernel()
    if not np.isfinite(gradient).all():
        raise RuntimeError("independent gradient is nonfinite")
    record = {
        "schema": "vibeqc.ccsd.gradient_reference",
        "schema_version": 1,
        "pyscf": pyscf.__version__,
        "inputs": value,
        "nmo": mol.nao_nr(),
        "hf_energy": float(mf.e_tot),
        "correlation_energy": float(coupled.e_corr),
        "total_energy": float(coupled.e_tot),
        "gradient": gradient.tolist(),
        "units": {"energy": "Eh", "gradient": "Eh/bohr", "coordinates": "bohr"},
        "scf": {
            "conv_tol": mf.conv_tol,
            "conv_tol_grad": mf.conv_tol_grad,
            "direct_scf_tol": mf.direct_scf_tol,
            "max_cycle": mf.max_cycle,
        },
        "ccsd": {
            "conv_tol": coupled.conv_tol,
            "conv_tol_normt": coupled.conv_tol_normt,
            "max_cycle": coupled.max_cycle,
            "frozen": 0,
        },
        "cphf": {"tol": 1e-13, "max_cycle": 100},
        "default_cphf_gradient_difference": float(
            np.max(abs(default_gradient - gradient))
        ),
        "gradient_source_sha256": hashlib.sha256(
            Path(inspect.getfile(grad_ccsd)).read_bytes()
        ).hexdigest(),
        "note": "CPHF numerical controls tightened; no independent equations modified",
    }
    record["content_hash"] = canonical_hash(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--case", choices=CASES, action="append")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.case or CASES:
        record = generate(name)
        (args.output / f"{name}.json").write_text(
            json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        print(
            name,
            record["total_energy"],
            record["default_cphf_gradient_difference"],
            flush=True,
        )


if __name__ == "__main__":
    main()
