"""Small explicit gradient fixtures and pinned independent reference records."""

import json
import typing
from copy import deepcopy
from pathlib import Path

from tools.cc_endpoint_fixtures import load as load_endpoint
from tools.cc_endpoint_fixtures import source_arguments as endpoint_source_arguments
from tools.vibeqc_validation.schema import canonical_hash

ROOT = Path(__file__).resolve().parents[1] / "tests/reference_data/cc/gradients"
CASES = ("h2", "h2_shifted", "h2o", "nh3", "ch4", "h2_d_cartesian", "h2_d_spherical")


def inputs(name: typing.Any) -> typing.Any:
    if name not in CASES:
        raise ValueError("unknown small CCSD gradient case")
    base = "h2" if name.startswith("h2_") else name
    meta, _ = load_endpoint(base)
    value = deepcopy(meta["inputs"])
    value["name"] = name
    if name == "h2_shifted":
        value["coordinates"][1][2] += 0.15
    if name.startswith("h2_d_"):
        value["shells"].append(
            {"atom_index": 0, "angular_momentum": 2, "primitives": [[0.6, 1.0]]}
        )
        value["shells"].sort(key=lambda s: (s["atom_index"], s["angular_momentum"]))
        value["basis_representation"] = name.removeprefix("h2_d_")
        value["basis_name"] = "STO-3G plus explicit first-center d(0.6)"
    return value


def source_arguments(value: typing.Any) -> typing.Any:
    """Validate the explicit molecular schema without dropping unsupported physics."""
    expected = set(inputs("h2"))
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(
            "CC gradient input must use the complete supported molecular schema"
        )
    if (
        value["kind"] != "molecule"
        or value["method"] != "rhf"
        or value["multiplicity"] != 1
    ):
        raise ValueError("CC gradient input requires a closed-shell RHF molecule")
    if value["auxiliary_centers"]:
        raise ValueError("CC gradient input does not support an auxiliary basis")
    settings, conventions = value["cc_settings"], value["conventions"]
    if (
        not isinstance(settings, dict)
        or settings.get("method") != "CCSD"
        or type(settings.get("frozen_core")) is not int
        or settings["frozen_core"] != 0
    ):
        raise ValueError(
            "CC gradient input supports only unfrozen CCSD, not (T) or frozen core"
        )
    if (
        not isinstance(conventions, dict)
        or conventions.get("length_unit") != "bohr"
        or conventions.get("hamiltonian")
        != "all-electron nonrelativistic Coulomb; no ECP"
        or conventions.get("frozen_core") != 0
    ):
        raise ValueError(
            "CC gradient input requires explicit Bohr all-electron conventions"
        )
    if value["basis_representation"] not in ("cartesian", "spherical"):
        raise ValueError("unsupported CC gradient basis representation")
    args = endpoint_source_arguments(value)
    args["representation"] = value["basis_representation"]
    args["multiplicity"] = value["multiplicity"]
    return args


def load(name: typing.Any, root: typing.Any = ROOT) -> typing.Any:
    data = json.loads((Path(root) / f"{name}.json").read_text())
    if (
        data["schema"] != "vibeqc.ccsd.gradient_reference"
        or data["schema_version"] != 1
        or data["pyscf"] != "2.14.0"
    ):
        raise ValueError("unsupported CCSD gradient reference schema/version")
    digest = data.pop("content_hash")
    if canonical_hash(data) != digest or data["inputs"] != inputs(name):
        raise ValueError("CCSD gradient reference content/input identity mismatch")
    data["content_hash"] = digest
    return data
