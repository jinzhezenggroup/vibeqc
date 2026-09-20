"""Validate native CPU UKS against independent matched-grid PySCF/Libxc SCF.

PySCF owns its AO integrals, Coulomb build, XC integration, SCF iteration and
final residual evaluation. VibeQC and PySCF receive the same atoms, basis,
charge, spin, GridSpec-v1 points and quadrature weights. This is a small
endpoint gate, not a quadrature-convergence, gradient, batch or CUDA claim.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import typing
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import numpy as np
from vibeqc import Atom
from vibeqc.calculator import Calculator, _named_basis_shells
from vibeqc_compiler.dft import GridSpec, MolecularGrid

from tools.generate_validation_references import pyscf_molecule

CASES = (
    {
        "name": "h2_minus_doublet",
        "atoms": (("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))),
        "charge": -1,
        "multiplicity": 2,
    },
    {
        "name": "h2_plus_fully_polarized",
        "atoms": (("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))),
        "charge": 1,
        "multiplicity": 2,
    },
    {
        # LDA can cycle between symmetry-related pi occupations even though
        # its physical energy and residual are stationary. Exercise the public
        # convergence result as well as an independent endpoint for this case.
        "name": "oh_doublet",
        "atoms": (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8))),
        "charge": 0,
        "multiplicity": 2,
        "reference_symmetry": "C2v",
    },
)

METHODS = (
    ("lda-uks", "LDA_X,LDA_C_PW"),
    ("pbe-uks", "GGA_X_PBE,GGA_C_PBE"),
)


def digest(array: np.ndarray) -> str:
    return sha256(np.asarray(array, dtype=np.float64, order="C").tobytes()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def pyscf_input(case: dict) -> tuple[dict, tuple[Atom, ...]]:
    atoms = tuple(Atom.from_value(value) for value in case["atoms"])
    shells = _named_basis_shells("sto-3g", atoms)
    inputs = {
        "atomic_numbers": [atom.atomic_number for atom in atoms],
        "coordinates": [atom.position for atom in atoms],
        "shells": [
            {
                "atom_index": shell.atom_index,
                "angular_momentum": shell.angular_momentum,
                "primitives": [
                    (primitive.exponent, primitive.coefficient)
                    for primitive in shell.primitives
                ],
            }
            for shell in shells
        ],
        "basis_representation": "cartesian",
        "charge": case["charge"],
        "multiplicity": case["multiplicity"],
    }
    return inputs, atoms


def physical_residual_rms(
    fock: np.ndarray, density: np.ndarray, overlap: np.ndarray
) -> float:
    residuals = [
        fock[spin] @ density[spin] @ overlap - overlap @ density[spin] @ fock[spin]
        for spin in range(2)
    ]
    joined = np.concatenate([value.ravel() for value in residuals])
    return float(np.sqrt(np.mean(joined * joined)))


def independent_uks(
    inputs: dict, grid: typing.Any, xc_code: str, symmetry: str | None = None
) -> dict:
    import pyscf
    from pyscf import dft

    if pyscf.__version__ != "2.14.0":
        raise RuntimeError("UKS endpoint validation requires PySCF 2.14.0")
    libxc_version = dft.libxc.__version__
    if libxc_version != "7.0.0":
        raise RuntimeError("UKS endpoint validation requires Libxc 7.0.0")

    mol, _, angular_momenta = pyscf_molecule(inputs)
    if symmetry is not None:
        # Resolve OH's pi orientation in an exact molecular/grid subgroup.
        # The final gate still measures the full unrestricted AO commutator,
        # including rotations excluded by the symmetry-adapted iteration.
        mol.symmetry = symmetry
        mol.build()
    mf = dft.UKS(mol)
    mf.xc = xc_code
    mf.grids.coords = np.asarray(grid.points)
    mf.grids.weights = np.asarray(grid.weights)
    mf.grids.non0tab = None
    mf.small_rho_cutoff = 0.0
    mf.conv_tol = 1.0e-12
    mf.conv_tol_grad = 1.0e-10
    mf.max_cycle = 200
    mf.diis_space = 8
    mf.direct_scf_tol = 1.0e-14
    history = []
    mf.callback = lambda environment: history.append(
        {
            "iteration": int(environment["cycle"] + 1),
            "energy_hartree": float(environment["e_tot"]),
            "orbital_gradient_norm": float(environment["norm_gorb"]),
            "density_change_norm": float(environment["norm_ddm"]),
        }
    )
    energy = float(mf.kernel())
    density = np.asarray(mf.make_rdm1())
    hcore = np.asarray(mf.get_hcore())
    potential = np.asarray(mf.get_veff(mol, density))
    fock = hcore[None, :, :] + potential
    residual = physical_residual_rms(fock, density, np.asarray(mf.get_ovlp()))
    return {
        "energy_hartree": energy,
        "converged": bool(mf.converged),
        "initial_guess": mf.init_guess,
        "symmetry": symmetry,
        "iterations": len(history),
        "history": history,
        "physical_residual_rms": residual,
        "spin_square": [float(value) for value in mf.spin_square()],
        "angular_momenta": angular_momenta,
        "pyscf": pyscf.__version__,
        "libxc": libxc_version,
    }


def native_uks(case: dict, method: str) -> dict:
    result = Calculator(
        method=method,
        basis="sto-3g",
        device="cpu",
        max_iterations=200,
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
        diis_history=8,
    ).singlepoint(
        case["atoms"],
        charge=case["charge"],
        multiplicity=case["multiplicity"],
        properties=("energy",),
    )
    return {
        "energy_hartree": float(result.energy),
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "energy_change_hartree": float(result.energy_change),
        "density_change_rms": float(result.density_rms),
        "physical_residual_rms": float(result.physical_residual_rms),
        "executed_backend": result.executed_backend,
    }


def validate() -> dict:
    energy_gate = 1.0e-8
    residual_gate = 1.0e-9
    rows = []
    for case in CASES:
        inputs, atoms = pyscf_input(case)
        grid = MolecularGrid(
            atoms,
            GridSpec(),
            charge=case["charge"],
            multiplicity=case["multiplicity"],
        ).explicit()
        for method, xc_code in METHODS:
            native = native_uks(case, method)
            independent = independent_uks(
                inputs, grid, xc_code, case.get("reference_symmetry")
            )
            difference = abs(native["energy_hartree"] - independent["energy_hartree"])
            passed = (
                native["converged"]
                and independent["converged"]
                and native["executed_backend"] == "cpu_reference"
                and difference <= energy_gate
                and native["physical_residual_rms"] <= residual_gate
                and independent["physical_residual_rms"] <= residual_gate
            )
            rows.append(
                {
                    "case": case["name"],
                    "method": method,
                    "charge": case["charge"],
                    "multiplicity": case["multiplicity"],
                    "spin_occupations": [
                        (
                            sum(inputs["atomic_numbers"])
                            - case["charge"]
                            + case["multiplicity"]
                            - 1
                        )
                        // 2,
                        (
                            sum(inputs["atomic_numbers"])
                            - case["charge"]
                            - case["multiplicity"]
                            + 1
                        )
                        // 2,
                    ],
                    "grid": {
                        "identity": grid.provenance["grid_identity"],
                        "point_count": len(grid.points),
                        "points_sha256": digest(grid.points),
                        "weights_sha256": digest(grid.weights),
                    },
                    "native": native,
                    "independent": independent,
                    "absolute_energy_error_hartree": difference,
                    "passed": passed,
                }
            )

    return {
        "schema": "vibeqc.uks-endpoint-validation",
        "version": 1,
        "source": {
            "commit": git_output("rev-parse", "HEAD"),
            "status_porcelain": git_output("status", "--short"),
            "validator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "vibeqc_library": os.environ.get("VIBEQC_LIBRARY"),
            "threads": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "gates": {
            "absolute_energy_error_hartree": energy_gate,
            "physical_residual_rms": residual_gate,
        },
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
        "limitations": [
            "small STO-3G H2 and OH endpoints only",
            "fixed GridSpec-v1 quadrature; no grid-convergence claim",
            "CPU conventional-J energy only",
            "no gradients, density fitting, prepared batches, CUDA or performance claim",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    for row in record["rows"]:
        print(
            row["case"],
            row["method"],
            f"dE={row['absolute_energy_error_hartree']:.3e}",
            f"native_residual={row['native']['physical_residual_rms']:.3e}",
            f"pyscf_residual={row['independent']['physical_residual_rms']:.3e}",
            "PASS" if row["passed"] else "FAIL",
        )
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
