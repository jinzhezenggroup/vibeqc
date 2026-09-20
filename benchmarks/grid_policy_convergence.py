"""Independent production DFT grid convergence and point-cost qualification.

The gate intentionally does not call VibeQC's native KS implementation. PySCF
performs independent RKS SCF and analytic grid-response gradients on explicit
VibeQC quadratures. A 96x32x64 reference is first checked against a denser
120x40x80 grid before standard/tight profiles are admitted against it.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
import typing
from dataclasses import asdict
from pathlib import Path

import numpy as np
from vibeqc import Atom
from vibeqc_compiler.dft.grid import GridPolicy, GridSpec, MolecularGrid

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path

ATOMS = (
    ("O", (0.1, -0.1, 0.0)),
    ("H", (0.1, 0.2, 1.7)),
    ("H", (1.6, -0.2, -0.5)),
)
DENSE_SHAPE = (96, 32, 64)
ULTRA_SHAPE = (120, 40, 80)

GATES = {
    "dense_reference": {"energy": 2.0e-8, "force": 1.0e-6},
    "lda-standard": {"energy": 1.0e-6, "force": 5.0e-5, "max_point_fraction": 0.20},
    "lda-tight": {"energy": 5.0e-7, "force": 1.5e-5, "max_point_fraction": 0.45},
    "pbe-standard": {"energy": 1.5e-6, "force": 3.0e-5, "max_point_fraction": 0.20},
    "pbe-tight": {"energy": 5.0e-7, "force": 5.0e-6, "max_point_fraction": 0.45},
}


def _explicit_arrays(
    atoms: tuple[Atom, ...], spec: GridSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = MolecularGrid(atoms, spec=spec)
    points: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    owners: list[int] = []
    raw_points: list[np.ndarray] = []
    raw_weights: list[np.ndarray] = []
    raw_owners: list[int] = []
    for tile in grid.tiles(tile_points=16384):
        points.append(tile.points)
        weights.append(tile.weights)
        owners.extend(tile.owners)
    for tile in grid._raw_tiles(tile_points=16384):
        raw_points.append(tile.points)
        raw_weights.append(tile.weights)
        raw_owners.extend(tile.owners)
    return (
        np.concatenate(points),
        np.concatenate(weights),
        np.asarray(owners),
        np.concatenate(raw_points),
        np.concatenate(raw_weights),
        np.asarray(raw_owners),
    )


def _endpoint(spec: GridSpec, xc: str) -> dict[str, typing.Any]:
    from pyscf import dft, gto, lib

    atoms = tuple(Atom.from_value(atom) for atom in ATOMS)
    labels = [f"{symbol}{index}" for index, (symbol, _) in enumerate(ATOMS)]
    molecule = gto.M(
        atom=[(label, xyz) for label, (_, xyz) in zip(labels, ATOMS, strict=True)],
        basis="sto-3g",
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    points, weights, _owners, raw_points, raw_weights, raw_owners = _explicit_arrays(
        atoms, spec
    )
    mean_field = dft.RKS(molecule)
    mean_field.xc = xc
    mean_field.grids.coords = points
    mean_field.grids.weights = weights
    mean_field.grids.radii_adjust = None
    mean_field.grids.gen_atomic_grids = lambda *args, **kwargs: {
        molecule.atom_symbol(atom): (
            raw_points[raw_owners == atom] - molecule.atom_coord(atom),
            raw_weights[raw_owners == atom],
        )
        for atom in range(molecule.natm)
    }
    mean_field.small_rho_cutoff = 0
    mean_field.conv_tol = 1.0e-13
    mean_field.conv_tol_grad = 1.0e-10
    mean_field.max_cycle = 150
    lib.num_threads(1)
    started = time.perf_counter()
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError("independent PySCF grid-convergence SCF did not converge")
    gradient = mean_field.nuc_grad_method()
    gradient.grid_response = True
    value = gradient.kernel()
    return {
        "energy_hartree": float(mean_field.e_tot),
        "gradient_hartree_per_bohr": value.tolist(),
        "npoint": len(points),
        "seconds": time.perf_counter() - started,
    }


def _error(
    candidate: dict[str, typing.Any], reference: dict[str, typing.Any]
) -> dict[str, float]:
    return {
        "energy_hartree": abs(
            candidate["energy_hartree"] - reference["energy_hartree"]
        ),
        "force_hartree_per_bohr": float(
            np.max(
                np.abs(
                    np.asarray(candidate["gradient_hartree_per_bohr"])
                    - np.asarray(reference["gradient_hartree_per_bohr"])
                )
            )
        ),
    }


def _spec(
    shape: tuple[int, int, int], radii: tuple[tuple[int, float], ...]
) -> GridSpec:
    radial, polar, azimuth = shape
    return GridSpec(
        version=2,
        radial_points=radial,
        angular_polar=polar,
        angular_azimuth=azimuth,
        partition_iterations=3,
        element_radii=radii,
    )


def qualify() -> dict[str, typing.Any]:
    import pyscf

    radii = GridPolicy().resolve("pbe-rks").element_radii
    dense_spec = _spec(DENSE_SHAPE, radii)
    ultra_spec = _spec(ULTRA_SHAPE, radii)
    result: dict[str, typing.Any] = {
        "schema": "vibeqc.production-grid-convergence",
        "version": 1,
        "molecule": {"atoms_bohr": ATOMS, "basis": "sto-3g"},
        "pyscf_version": pyscf.__version__,
        "numpy_version": np.__version__,
        "gates": GATES,
        "profiles": {},
        "reference_stability": {},
    }
    references: dict[str, dict[str, typing.Any]] = {}
    for family, xc in (("lda", "LDA_X,LDA_C_PW"), ("pbe", "PBE")):
        dense = _endpoint(dense_spec, xc)
        ultra = _endpoint(ultra_spec, xc)
        stability = _error(dense, ultra)
        result["reference_stability"][family] = {
            "dense_spec": asdict(dense_spec),
            "ultra_spec": asdict(ultra_spec),
            "dense": dense,
            "ultra": ultra,
            "error": stability,
            "passed": (
                stability["energy_hartree"] <= GATES["dense_reference"]["energy"]
                and stability["force_hartree_per_bohr"]
                <= GATES["dense_reference"]["force"]
            ),
        }
        references[family] = dense

    production_specs = {
        "lda-standard": GridPolicy().resolve("lda-rks"),
        "lda-tight": GridPolicy("tight").resolve("lda-rks"),
        "pbe-standard": GridPolicy().resolve("pbe-rks"),
        "pbe-tight": GridPolicy("tight").resolve("pbe-rks"),
        "pbe-legacy48": _spec((48, 16, 32), radii),
    }
    for name, spec in production_specs.items():
        family = name.split("-", 1)[0]
        xc = "LDA_X,LDA_C_PW" if family == "lda" else "PBE"
        endpoint = _endpoint(spec, xc)
        error = _error(endpoint, references[family])
        profile = {
            "spec": asdict(spec),
            "endpoint": endpoint,
            "error_vs_dense": error,
            "point_fraction_of_dense": endpoint["npoint"]
            / references[family]["npoint"],
        }
        gate = GATES.get(name)
        if gate is not None:
            profile["passed"] = (
                error["energy_hartree"] <= gate["energy"]
                and error["force_hartree_per_bohr"] <= gate["force"]
                and profile["point_fraction_of_dense"] <= gate["max_point_fraction"]
            )
        result["profiles"][name] = profile

    pbe_standard = result["profiles"]["pbe-standard"]
    pbe_legacy = result["profiles"]["pbe-legacy48"]
    result["accuracy_cost_checks"] = {
        "pbe_standard_improves_legacy48_energy": (
            pbe_standard["error_vs_dense"]["energy_hartree"]
            < pbe_legacy["error_vs_dense"]["energy_hartree"]
        ),
        "pbe_standard_improves_legacy48_force": (
            pbe_standard["error_vs_dense"]["force_hartree_per_bohr"]
            < pbe_legacy["error_vs_dense"]["force_hartree_per_bohr"]
        ),
        "lda_tight_improves_standard_force": (
            result["profiles"]["lda-tight"]["error_vs_dense"]["force_hartree_per_bohr"]
            < result["profiles"]["lda-standard"]["error_vs_dense"][
                "force_hartree_per_bohr"
            ]
        ),
        "pbe_tight_improves_standard_force": (
            result["profiles"]["pbe-tight"]["error_vs_dense"]["force_hartree_per_bohr"]
            < pbe_standard["error_vs_dense"]["force_hartree_per_bohr"]
        ),
        "pbe_standard_point_cost_vs_legacy48": (
            pbe_standard["endpoint"]["npoint"] / pbe_legacy["endpoint"]["npoint"]
        ),
    }
    failures = [
        f"reference/{family}"
        for family, row in result["reference_stability"].items()
        if not row["passed"]
    ]
    failures.extend(
        f"profile/{name}"
        for name, row in result["profiles"].items()
        if "passed" in row and not row["passed"]
    )
    failures.extend(
        f"accuracy_cost/{name}"
        for name, passed in result["accuracy_cost_checks"].items()
        if isinstance(passed, bool) and not passed
    )
    result["failures"] = failures
    result["passed"] = not failures
    if failures:
        raise AssertionError(f"production grid convergence failed: {failures}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=raw_output_path)
    args = parser.parse_args()
    result = qualify()
    try:
        root = Path(__file__).resolve().parents[1]
        result["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        result["git_status"] = subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        result["git_head"] = None
        result["git_status"] = []
    result["platform"] = platform.platform()
    text = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
