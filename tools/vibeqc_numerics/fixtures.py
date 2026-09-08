"""Whole-family accuracy splits and two additional independent HF references."""

from copy import deepcopy
from pathlib import Path

from vibeqc import Atom, Calculator

from tools.vibeqc_validation.fixtures import load_fixtures, molecular_inputs

REFERENCE_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "tests/reference_data/accuracy"
)
TRAINING_FAMILIES = ("water", "methane")
FAMILIES = {
    "h2": "hydrogen",
    "h2o": "water",
    "ch4": "methane",
    "nh3": "ammonia",
    "he": "helium",
    "hf-plus-uhf": "hydrogen-fluoride",
    "hf": "hydrogen-fluoride",
    "h2-def2-svp": "hydrogen",
}


def extra_inputs():
    """Hold out HF as a molecule and def2-SVP as a complete basis family."""
    originals = {row["name"]: row for row in molecular_inputs()}
    hf = deepcopy(originals["hf-plus-uhf"])
    hf.update(name="hf", charge=0, multiplicity=1, method="rhf", cc_settings=None)
    hydrogen = deepcopy(originals["h2"])
    hydrogen.update(name="h2-def2-svp", basis_name="def2-svp", cc_settings=None)
    atoms = tuple(
        Atom(z, tuple(x))
        for z, x in zip(
            hydrogen["atomic_numbers"], hydrogen["coordinates"], strict=True
        )
    )
    shells = Calculator(basis="def2-svp")._shells_for_atoms(atoms)
    hydrogen["shells"] = [
        {
            "atom_index": s.atom_index,
            "angular_momentum": s.angular_momentum,
            "primitives": [[p.exponent, p.coefficient] for p in s.primitives],
        }
        for s in shells
    ]
    return [hf, hydrogen]


def accuracy_suite():
    """Load only independent molecular references, with their pinned hashes."""
    return [
        r for r in load_fixtures() if r["inputs"]["kind"] == "molecule"
    ] + load_fixtures(REFERENCE_DIRECTORY)
