"""Export larger current-orbital D/C workloads with pinned independent PySCF.

Only the explicit quadrature/basis inputs are shared with production. PySCF
supplies the SCF states, AO features and complete fixed-grid XC energy/matrix.
The deliberately coarse radial rule stays within the audited XC interior;
no point is removed based on density, and no weight or occupation is repaired.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import numpy as np
from vibeqc import Atom, Primitive, Shell
from vibeqc.calculator import _named_basis_shells
from vibeqc_compiler.common.provenance import canonical_hash, file_hash
from vibeqc_compiler.dft.grid import GridSpec, MolecularGrid

from tools.generate_validation_references import pyscf_molecule
from tools.generate_xc_integration_references import FUNCTIONALS


def workloads():
    """Compact, extended and diffuse inputs with genuine RKS/UKS occupations."""
    water = ((8, (0, 0, 0)), (1, (0, -1.43, 1.11)), (1, (0, 1.43, 1.11)))
    cluster = []
    for x, y, z in (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ):
        cluster.extend(
            (
                element,
                tuple(v + 5.5 * d for v, d in zip(position, (x, y, z), strict=True)),
            )
            for element, position in water
        )
    oh = ((8, (0, 0, 0)), (1, (0.1, 0.2, 1.82)))
    return (
        ("water_svp", water, "def2-svp", 0, 1, False),
        ("water_tzvp", water, "def2-tzvp", 0, 1, False),
        (
            "co2_tzvp",
            ((8, (0, 0, -2.2)), (6, (0, 0, 0)), (8, (0, 0, 2.2))),
            "def2-tzvp",
            0,
            1,
            False,
        ),
        ("water4_svp", tuple(cluster[:12]), "def2-svp", 0, 1, False),
        ("water8_svp", tuple(cluster), "def2-svp", 0, 1, False),
        ("oh_diffuse", oh, "def2-svp", 0, 2, True),
        ("oh_minus_diffuse", oh, "def2-svp", -1, 1, True),
    )


def generate(directory):
    """Write immutable reference inputs plus independent full-grid E/V blocks."""
    import pyscf
    from pyscf import dft
    from pyscf.dft import gen_grid, libxc, numint

    if pyscf.__version__ != "2.14.0" or libxc.__version__ != "7.0.0":
        raise RuntimeError("requires PySCF 2.14.0 / Libxc 7.0.0")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = []
    # Radial order two is a declared fixed-input workload, not a claim of
    # converged molecular quadrature. Angular/atom replication grows npoint.
    grid_spec = GridSpec(2, 16, 32)
    for name, atom_values, basis_name, charge, multiplicity, diffuse in workloads():
        atoms = tuple(Atom.from_value(value) for value in atom_values)
        shells = list(_named_basis_shells(basis_name, atoms))
        if diffuse:
            for atom, value in enumerate(atoms):
                shells.extend(
                    Shell(
                        atom,
                        angular,
                        (Primitive(0.02 if value.atomic_number == 8 else 0.01, 1.0),),
                    )
                    for angular in ((0, 1) if value.atomic_number == 8 else (0,))
                )
            shells.sort(key=lambda shell: (shell.atom_index, shell.angular_momentum))
        inputs = {
            "name": name,
            "basis_name": basis_name,
            "atomic_numbers": [a.atomic_number for a in atoms],
            "coordinates": [a.position for a in atoms],
            "basis_representation": "real_spherical",
            "charge": charge,
            "multiplicity": multiplicity,
            "shells": [
                {
                    "atom_index": s.atom_index,
                    "angular_momentum": s.angular_momentum,
                    "primitives": [(p.exponent, p.coefficient) for p in s.primitives],
                }
                for s in shells
            ],
            "grid_spec": asdict(grid_spec),
        }
        mol, scale, angular = pyscf_molecule(inputs)
        mf = (dft.RKS if multiplicity == 1 else dft.UKS)(mol)
        mf.xc = "GGA_X_PBE,GGA_C_PBE"
        mf.grids.level = 1
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-10, 1e-7, 100
        started = perf_counter()
        mf.kernel()
        producer_seconds = perf_counter() - started
        if not mf.converged:
            raise RuntimeError(f"independent SCF did not converge: {name}")
        dm = mf.make_rdm1()
        rks = multiplicity == 1
        matrices = np.stack((0.5 * dm, 0.5 * dm)) if rks else dm
        coefficients = (mf.mo_coeff, mf.mo_coeff) if rks else mf.mo_coeff
        occupations = (0.5 * mf.mo_occ, 0.5 * mf.mo_occ) if rks else mf.mo_occ
        grid = MolecularGrid(
            atoms, grid_spec, charge=charge, multiplicity=multiplicity
        ).explicit()
        reference_grid = gen_grid.Grids(mol)
        reference_grid.coords = np.array(grid.points)
        reference_grid.weights = np.array(grid.weights)
        reference_grid.non0tab, reference_grid.cutoff = None, 1e-30
        ni = numint.NumInt()
        ni.cutoff = 1e-30
        arrays = {
            "points": grid.points,
            "weights": grid.weights,
            "owners": np.asarray(grid.owners, dtype=np.uint32),
            "density_spin": matrices / (scale[:, None] * scale[None, :]),
            "ao_scale": scale,
        }
        for spin in range(2):
            occupied = occupations[spin] > 0
            arrays[f"coefficients_{spin}"] = (
                coefficients[spin][:, occupied] / scale[:, None]
            )
            arrays[f"occupations_{spin}"] = occupations[spin][occupied]
        # These independent Libcint/NumInt blocks audit all requested feature
        # entries on a distributed sample; full E/V below uses every point.
        sample_ids = np.linspace(0, len(grid.points) - 1, 64, dtype=int)
        ao = ni.eval_ao(mol, grid.points[sample_ids], deriv=1)
        arrays["sample_ids"] = sample_ids.astype(np.uint32)
        arrays["sample_jets"] = ao * scale
        arrays["sample_rho_gradient"] = np.stack(
            [ni.eval_rho(mol, ao, matrix, xctype="GGA", hermi=1) for matrix in matrices]
        )
        for label, code in FUNCTIONALS.items():
            run = ni.nr_rks if rks else ni.nr_uks
            electrons, energy, potential = run(mol, reference_grid, code, dm, hermi=1)
            arrays[label + "_energy"] = np.asarray([energy])
            arrays[label + "_electrons"] = np.atleast_1d(electrons)
            arrays[label + "_potential"] = potential * scale[:, None] * scale[None, :]
        arrays = {key: np.ascontiguousarray(value) for key, value in arrays.items()}
        metadata = {
            "schema": "vibeqc.density-workload-reference.v1",
            "inputs": inputs,
            "inputs_hash": canonical_hash(inputs),
            "grid_identity": grid.identity,
            "grid_provenance": json.loads(grid._provenance_json),
            "layout": "total" if rks else "separate_spin",
            "reference": {
                "PySCF": pyscf.__version__,
                "Libxc": libxc.__version__,
                "exporter_sha256": file_hash(Path(__file__)),
                "basis_adapter_sha256": file_hash(
                    ROOT / "tools/generate_validation_references.py"
                ),
                "numint_sha256": file_hash(Path(numint.__file__)),
                "angular_momenta": angular,
                "producer": "PySCF PBE RKS" if rks else "PySCF PBE UKS",
                "converged": bool(mf.converged),
                "energy": float(mf.e_tot),
                "gradient_norm": float(
                    np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ))
                ),
                "seconds": producer_seconds,
                "threads": pyscf.lib.num_threads(),
                "conv_tol": mf.conv_tol,
                "conv_tol_grad": mf.conv_tol_grad,
                "producer_grid_level": mf.grids.level,
                "endpoint_scope": "complete fixed-grid E/V on declared coarse radial quadrature; no filtering, clipping or grid convergence claim",
            },
            "arrays": {
                key: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": sha256(value.tobytes()).hexdigest(),
                }
                for key, value in arrays.items()
            },
        }
        metadata["identity"] = canonical_hash(metadata)
        np.savez_compressed(directory / f"{name}.npz", **arrays)
        (directory / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        manifest.append(
            {
                "name": name,
                "identity": metadata["identity"],
                "archive_sha256": file_hash(directory / f"{name}.npz"),
            }
        )
        print(name, mol.nao_nr(), len(grid.points), "SCF", producer_seconds, flush=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    generate(parser.parse_args().directory)
