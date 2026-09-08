"""Export exact-grid AO jets/features using pinned PySCF 2.14.0/libcint.

Original exporter; PySCF Apache-2.0 and libcint BSD-2-Clause are test-only
dependencies. No angular tables, solver implementation or production calls
are copied from them. Fixtures use arbitrary supplied spin density matrices.
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
import numpy as np
from vibeqc import Atom, Primitive, Shell
from vibeqc.calculator import _named_basis_shells
from vibeqc.profiles import canonical_hash

from tools.generate_validation_references import pyscf_molecule
from tools.vibeqc_dft.grid import GridSpec, MolecularGrid


def generate(directory):
    """Retain all derivative entries, grid data, density/orbital factors and hashes."""
    import pyscf
    from pyscf.dft import gen_grid, numint

    if pyscf.__version__ != "2.14.0":
        raise RuntimeError("reference export requires PySCF 2.14.0")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pair = (("He", (0, 0, 0)), ("H", (0.7, -0.2, 1.5)))
    mixed = (
        Shell(0, 0, (Primitive(0.6, 1),)),
        Shell(0, 1, (Primitive(0.2, 0.8), Primitive(0.8, 0.3))),
        Shell(1, 2, (Primitive(1.2, 1),)),
        Shell(1, 3, (Primitive(0.4, 1), Primitive(1.1, -0.1))),
    )
    fixtures = [
        ("h2", (("H", (0, 0, 0)), ("H", (0.1, 0.2, 1.4))), "sto-3g", "cartesian", 1),
        (
            "water",
            (("O", (0, 0, 0)), ("H", (1.4, 0.1, 1.1)), ("H", (-1.2, 0.2, 1.3))),
            "sto-3g",
            "cartesian",
            1,
        ),
        ("f_cartesian", pair, mixed, "cartesian", 2),
        ("f_spherical", pair, mixed, "real_spherical", 2),
        (
            "diffuse",
            pair,
            (Shell(0, 0, (Primitive(0.008, 1),)), Shell(1, 1, (Primitive(0.04, 1),))),
            "cartesian",
            2,
        ),
        (
            "tight",
            pair,
            (Shell(0, 0, (Primitive(1200.0, 1),)), Shell(1, 2, (Primitive(300.0, 1),))),
            "cartesian",
            2,
        ),
    ]
    manifest = []
    for name, atom_values, basis, representation, multiplicity in fixtures:
        atoms = tuple(Atom.from_value(a) for a in atom_values)
        shells = _named_basis_shells(basis, atoms) if isinstance(basis, str) else basis
        inputs = {
            "name": name,
            "atomic_numbers": [a.atomic_number for a in atoms],
            "coordinates": [a.position for a in atoms],
            "shells": [
                {
                    "atom_index": s.atom_index,
                    "angular_momentum": s.angular_momentum,
                    "primitives": [(p.exponent, p.coefficient) for p in s.primitives],
                }
                for s in shells
            ],
            "basis_representation": representation,
            "charge": 0,
            "multiplicity": multiplicity,
        }
        mol, scale, actual = pyscf_molecule(inputs)
        topology = MolecularGrid(atoms, GridSpec(2, 2, 4), multiplicity=multiplicity)
        grid = topology.explicit()
        # Give PySCF identical unpartitioned atomic rules and compare its native
        # Becke implementation. No agreement between distinct prescriptions is
        # inferred, and the production grid never calls this external routine.
        relative = (
            topology.radii[:, None, None] * topology.directions[None, :, :]
        ).reshape(-1, 3)
        raw_weights = (
            topology.radial_weights[:, None] * topology.angular_weights[None, :]
        ).ravel()
        tab = {mol.atom_symbol(a): (relative, raw_weights) for a in range(mol.natm)}
        independent_points, independent_weights = gen_grid.get_partition(
            mol, tab, radii_adjust=None
        )
        np.testing.assert_allclose(
            independent_points, grid.points, atol=1e-14, rtol=1e-14
        )
        np.testing.assert_allclose(
            independent_weights, grid.weights, atol=1e-11, rtol=1e-10
        )
        grid.write(directory / f"{name}-grid.json")
        rng = np.random.default_rng(160)
        close = np.array([a.position for a in atoms])[:, None, :] + np.array(
            [[0, 0, 0], [0.001, -0.002, 0.003], [0.04, 0.03, -0.02]]
        )
        points = np.concatenate(
            (
                grid.points,
                close.reshape(-1, 3),
                rng.normal(size=(19, 3)),
                [[30, -20, 10]],
            )
        )
        raw = mol.eval_gto(
            "GTOval_cart_deriv3" if mol.cart else "GTOval_sph_deriv3", points
        )
        ao = raw * scale
        n = mol.nao_nr()
        norb = min(4, n)
        c = rng.normal(size=(2, n, norb)) / np.sqrt(n)
        occupations = rng.uniform(0.1, 0.9, size=(2, norb))
        density = np.einsum("smi,si,sni->smn", c, occupations, c)
        features = np.stack(
            [
                numint.eval_rho(
                    mol,
                    raw[:10],
                    d * scale[:, None] * scale[None, :],
                    xctype="MGGA",
                    hermi=1,
                    with_lapl=True,
                )
                for d in density
            ]
        )
        gradient = features[:, 1:4].transpose(0, 2, 1)
        arrays = {
            "points": points,
            "ao_jets": ao,
            "density": density,
            "coefficients": c,
            "partitioned_grid_points": independent_points,
            "partitioned_grid_weights": independent_weights,
            "occupations": occupations,
            "rho": features[:, 0],
            "gradient": gradient,
            "tau": features[:, 5],
            "laplacian": features[:, 4],
            "sigma": np.stack(
                (
                    np.sum(gradient[0] ** 2, axis=1),
                    np.sum(gradient[0] * gradient[1], axis=1),
                    np.sum(gradient[1] ** 2, axis=1),
                )
            ),
            "overlap": mol.intor("int1e_ovlp") * scale[:, None] * scale[None, :],
        }
        arrays = {
            k: np.asarray(v, dtype=np.float64, order="C") for k, v in arrays.items()
        }
        metadata = {
            "schema": "vibeqc.grid-reference",
            "version": 1,
            "inputs": inputs,
            "inputs_hash": canonical_hash(inputs),
            "grid_hash": grid.identity,
            "reference": {
                "PySCF": pyscf.__version__,
                "NumPy": np.__version__,
                "licenses": {"PySCF": "Apache-2.0", "libcint": "BSD-2-Clause"},
                "exporter": "original tools/generate_grid_references.py",
                "actual_basis": actual,
            },
            "conventions": {
                "jets": "ordinary spatial derivatives; libcint order; no factorials",
                "density": "separate alpha/beta; arbitrary PSD, not SCF",
                "tau": "one-half gradient square",
                "sigma": "aa,ab,bb; no cross factor",
                "grid": "identical explicit points, not a quadrature-convergence comparison",
            },
            "arrays": {
                k: {"shape": list(v.shape), "sha256": sha256(v.tobytes()).hexdigest()}
                for k, v in arrays.items()
            },
        }
        np.savez_compressed(directory / f"{name}.npz", **arrays)
        (directory / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        manifest.append({"name": name, "arrays": metadata["arrays"]})
        print(name, n, len(points), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "tests/reference_data/grid"
    )
    generate(parser.parse_args().output)
