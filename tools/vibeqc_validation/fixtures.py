"""Pinned mathematical conventions and small deterministic reference inputs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from .schema import canonical_hash

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIRECTORY = ROOT / "tests/reference_data/validation"
REFERENCE_VERSION = 1
CONVENTIONS = {
    "version": 1,
    "length_unit": "bohr",
    "energy_unit": "hartree",
    "gradient_unit": "hartree/bohr",
    "force_sign": "force=-dE/dR",
    "cartesian_order": "descending lx, then descending ly; lz=l-lx-ly",
    "cartesian_normalization": "each contracted Cartesian component has unit self-overlap",
    "spherical_order": "libcint real harmonics; p=(x,y,z); l>=2 m=-l,...,+l",
    "spherical_normalization": "unit self-overlap; libcint phases",
    "eri_order": "chemist (ij|kl); physicist <ij|kl> = chemist (ik|jl)",
    "eri_storage": "C order, last AO index fastest",
    "derivative_order": "shell-center slot, xyz, i,j,k,l; nuclear-center derivative",
    "rhf_density": "P=2*C_occ*C_occ.T; F=h+J[P]-K[P]/2",
    "uhf_density": "P_alpha=C_alpha_occ*C_alpha_occ.T; P_beta likewise",
    "orbital_representation": "real spatial orbitals; alpha/beta separate for UHF",
    "spin_orbital_order": "when expanded: all alpha spatial orbitals, then all beta",
    "frozen_core": 0,
    "hamiltonian": "all-electron nonrelativistic Coulomb; no ECP",
    "orbital_order": "ascending canonical orbital energy within each spin",
    "orbital_phase": "largest-absolute AO coefficient positive; first index wins ties; no degenerate-subspace rotation",
    "nuclear_charges": "explicit atomic_numbers, never inferred from basis labels",
    "auxiliary_centers": "explicit independent identities and parent atom indices; [] means none",
}
SCF_SETTINGS = {
    "energy_tolerance": 1e-13,
    "gradient_tolerance": 1e-11,
    "max_iterations": 200,
    "initial_guess": "minao",
    "direct_scf_tol": 1e-14,
}


def molecular_inputs() -> list[dict]:
    """Use exact bundled coefficients and asymmetric, non-optimized geometries.

    HF+ is the additional open-shell hydrogen-fluoride UHF fixture. NH3 has
    enough occupied and virtual orbitals for a nonzero (T) contribution.
    Actual shell angular momenta are serialized, not guessed from basis names.
    """
    cases = [
        ("h2", [1, 1], [[0.13, -0.21, -0.67], [-0.08, 0.17, 0.74]], 0, 1),
        ("he", [2], [[0.17, -0.23, 0.31]], 0, 1),
        (
            "h2o",
            [8, 1, 1],
            [[0.1, -0.2, 0.03], [0.02, -1.61, 1.17], [0.19, 1.24, 1.08]],
            0,
            1,
        ),
        (
            "nh3",
            [7, 1, 1, 1],
            [
                [0.07, -0.11, 0.16],
                [1.72, 0.14, -0.54],
                [-0.68, 1.61, -0.42],
                [-0.93, -1.48, -0.61],
            ],
            0,
            1,
        ),
        (
            "ch4",
            [6, 1, 1, 1, 1],
            [
                [0.03, -0.04, 0.1],
                [1.21, 1.17, 1.31],
                [-1.12, -1.29, 1.18],
                [1.16, -1.23, -1.14],
                [-1.28, 1.22, -1.19],
            ],
            0,
            1,
        ),
        ("hf-plus-uhf", [9, 1], [[-0.13, 0.22, -0.74], [0.18, -0.09, 1.12]], 1, 2),
    ]
    pack = json.loads((ROOT / "python/vibeqc/data/basis_pack.json").read_text())
    elements = pack["bases"]["sto-3g"]["elements"]
    rows = []
    for name, charges, coordinates, charge, multiplicity in cases:
        shells = []
        for atom, z in enumerate(charges):
            for shell in elements[str(z)]:
                shells.append(
                    {
                        "atom_index": atom,
                        "angular_momentum": shell["angular_momentum"],
                        "primitives": [
                            [float(e), float(c)]
                            for e, c in zip(
                                shell["exponents"], shell["coefficients"], strict=True
                            )
                        ],
                    }
                )
        rows.append(
            {
                "name": name,
                "kind": "molecule",
                "conventions": CONVENTIONS,
                "atomic_numbers": charges,
                "coordinates": coordinates,
                "charge": charge,
                "multiplicity": multiplicity,
                "method": "rhf" if multiplicity == 1 else "uhf",
                "basis_name": "sto-3g",
                "basis_representation": "cartesian",
                "shells": shells,
                "auxiliary_centers": [],
                "scf_settings": SCF_SETTINGS,
                "cc_settings": {
                    "method": "CCSD(T)",
                    "frozen_core": 0,
                    "conv_tol": 1e-13,
                    "conv_tol_normt": 1e-11,
                    "max_cycle": 200,
                }
                if name == "nh3"
                else None,
            }
        )
    return deepcopy(rows)


def small_inputs(seed: int = 138) -> list[dict]:
    """Scale-controlled s/p/d/f quartets, including coincident shell centers.

    All exponents lie in [0.4, 2.0]; positive coefficients avoid nearly singular
    contraction normalization. The seed and explicit coefficients are both
    saved. This is a minimal fixture layer, not a second #135 f-shell matrix.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    rows = []
    for name, angular, mode in [
        ("ssss-asymmetric", [0, 0, 0, 0], "distinct"),
        ("psss-coincident", [1, 0, 0, 0], "coincident"),
        ("dpss-asymmetric", [2, 1, 0, 0], "distinct"),
        ("fsss-asymmetric", [3, 0, 0, 0], "distinct"),
        ("fsss-same-center", [3, 0, 0, 0], "same"),
        ("fsss-spherical", [3, 0, 0, 0], "distinct"),
    ]:
        centers = rng.uniform(-0.8, 0.8, (4, 3))
        if mode == "coincident":
            centers[1] = centers[0]
        if mode == "same":
            centers[:] = centers[0]
        shells = [
            {
                "atom_index": i,
                "angular_momentum": l,
                "primitives": np.column_stack(
                    (rng.uniform(0.4, 2.0, 1 + i % 3), rng.uniform(0.2, 1.0, 1 + i % 3))
                ).tolist(),
            }
            for i, l in enumerate(angular)
        ]
        rows.append(
            {
                "name": name,
                "kind": "quartet",
                "conventions": CONVENTIONS,
                "seed": seed,
                "rng": "numpy.PCG64",
                "coordinates": centers.tolist(),
                "atomic_numbers": [1] * 4,
                "charge": 0,
                "multiplicity": 1,
                "basis_representation": "spherical"
                if name.endswith("spherical")
                else "cartesian",
                "shells": shells,
                "auxiliary_centers": [],
                "center_case": mode,
            }
        )
    return deepcopy(rows)


def mathematical_hash(inputs: dict) -> str:
    """Identify explicit mathematics independently of fixture labels or RNG history.

    Coefficients and coordinates are authoritative. Renaming a fixture, basis
    alias, or seed after saving those numbers must not create a second identity.
    Array order, center identities, occupations, and conventions remain hashed.
    """
    annotations = {"name", "basis_name", "seed", "rng", "center_case"}

    def numbers(value):
        # JSON 1 and 1.0 (also -0.0 and 0) describe the same mathematical
        # number. Convert exactly integral floats to integers without rounding
        # fractional values or passing large integer dimensions through float.
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, dict):
            return {k: numbers(v) for k, v in value.items()}
        if isinstance(value, list):
            return [numbers(v) for v in value]
        return value

    return canonical_hash(
        numbers({k: v for k, v in inputs.items() if k not in annotations})
    )


def validate_fixture(row: dict) -> None:
    """Reject changed AO/sign conventions, stale versions, or damaged data."""
    if (
        row.get("schema") != "vibeqc.reference"
        or row.get("schema_version") != REFERENCE_VERSION
    ):
        raise ValueError("unsupported reference version")
    if row["inputs"]["conventions"] != CONVENTIONS:
        raise ValueError(
            "reference AO ordering, normalization, or force conventions differ"
        )
    if mathematical_hash(row["inputs"]) != row["inputs_hash"]:
        raise ValueError("reference mathematical inputs hash mismatch")
    if canonical_hash(row["data"]) != row["data_hash"]:
        raise ValueError("reference data hash mismatch")
    if row["angular_momenta_loaded"] != [
        s["angular_momentum"] for s in row["inputs"]["shells"]
    ]:
        raise ValueError("loaded basis angular momenta mismatch")
    if row["inputs"]["kind"] == "molecule" and not np.allclose(
        row["data"]["forces"],
        -np.asarray(row["data"]["gradient"]),
        atol=1e-14,
        rtol=0,
    ):
        raise ValueError("reference force sign mismatch")
    provenance = row["provenance"]
    if provenance.get("schema_version") != 1 or not all(
        provenance.get(k)
        for k in ("pyscf", "numpy", "blas", "generator_sha256", "libcint_sha256")
    ):
        raise ValueError("missing or unsupported reference provenance version")
    if canonical_hash(provenance) != row["provenance_hash"]:
        raise ValueError("reference provenance/version hash mismatch")


def load_fixtures(directory: str | Path = REFERENCE_DIRECTORY) -> list[dict]:
    """Load a pinned manifest, checking reference bytes before numerical use."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != REFERENCE_VERSION:
        raise ValueError("unsupported reference manifest version")
    rows = []
    for item in manifest["fixtures"]:
        row = json.loads((directory / item["file"]).read_text())
        validate_fixture(row)
        if canonical_hash(row) != item["record_hash"]:
            raise ValueError("reference differs from pinned manifest")
        rows.append(row)
    return rows


def calculator_inputs(inputs: dict) -> dict:
    """Translate saved coefficients to VibeQC; never load a reference program."""
    from vibeqc import Primitive, Shell

    return {
        "method": inputs.get("method", "rhf"),
        "basis_representation": inputs["basis_representation"],
        "basis": tuple(
            Shell(
                s["atom_index"],
                s["angular_momentum"],
                tuple(Primitive(*p) for p in s["primitives"]),
            )
            for s in inputs["shells"]
        ),
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "screening_tolerance": 1e-14,
        "max_iterations": 200,
    }
