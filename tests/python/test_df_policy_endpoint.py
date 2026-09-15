"""Scientific reference admission must fail before permissive NumPy arithmetic."""

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.df_policy_endpoint import CASES, independent_reference

EVIDENCE = (
    Path(__file__).resolve().parents[2] / "benchmarks/results/issue377-379-df/gpu4pyscf"
)


def reference_inputs(aos):
    """Reuse the versioned independent results without loading a native backend."""
    reference = json.loads((EVIDENCE / f"{CASES[aos]}.json").read_text())
    metadata = json.loads(json.dumps(reference["vibeqc"]["cold_convergence"]))
    for row in metadata:
        for role in ("orbital", "auxiliary"):
            electrons = row["basis_metadata"][role]["electrons"]
            for key, value in electrons.items():
                if isinstance(value, list):
                    electrons[key] = tuple(value)
    return reference, (
        aos,
        np.asarray(reference["gpu4pyscf"]["energies_hartree"]),
        np.asarray(reference["gpu4pyscf"]["forces_hartree_per_bohr"]),
        [row["basis_metadata"] for row in metadata],
    )


@pytest.mark.parametrize("aos", CASES)
def test_retained_independent_references_match_current_workloads(aos):
    """All four retained geometries and scientific settings remain admissible."""
    reference, args = reference_inputs(aos)
    energy, forces = independent_reference(reference, *args)
    np.testing.assert_array_equal(energy, args[1])
    np.testing.assert_array_equal(forces, args[2])


@pytest.mark.parametrize(
    "key,value",
    [
        ("case", CASES[192]),
        ("ao_count", 192),
        ("batch_size", 2),
        ("basis_representation", "cartesian"),
        ("method", "uhf"),
        ("charge", 1),
        ("auxiliary_basis", "def2-universal-jfit"),
        ("density_fitting_relative_threshold", 1e-9),
        ("energy_tolerance", 1e-8),
    ],
)
def test_reference_rejects_changed_workload(key, value):
    """A matching case label cannot authorize a different scientific workload."""
    reference, args = reference_inputs(96)
    reference["workload"][key] = value
    with pytest.raises(RuntimeError, match=f"workload differs: {key}"):
        independent_reference(reference, *args)


@pytest.mark.parametrize("change", ["coordinate", "element", "basis"])
def test_reference_rejects_changed_geometry_or_basis(change):
    """Atom identity, coordinates and actual basis fingerprints are all checked."""
    reference, args = reference_inputs(96)
    if change == "coordinate":
        reference["workload"]["geometries"][0][0]["coordinates_bohr"][0] += 0.001
    elif change == "element":
        reference["workload"]["geometries"][0][0]["element"] = "N"
    else:
        # Copy the caller metadata so changing the retained basis cannot also
        # mutate the independently supplied actual metadata through an alias.
        args = (*args[:3], json.loads(json.dumps(args[3])))
        reference["vibeqc"]["cold_convergence"][0]["basis_metadata"]["orbital"][
            "basis_identity"
        ] = "stale-basis"
    with pytest.raises(RuntimeError, match="differs"):
        independent_reference(reference, *args)


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("energies_hartree", 0.0, "energy shape differs"),
        ("forces_hartree_per_bohr", [[[0.0, 0.0, 0.0]]], "force shape differs"),
        ("energies_hartree", [float("nan")], "energy is not finite"),
        (
            "forces_hartree_per_bohr",
            [[[float("inf"), 0.0, 0.0]] * 12],
            "force is not finite",
        ),
    ],
)
def test_reference_rejects_broadcastable_or_nonfinite_arrays(key, value, message):
    """Broadcasting and NaN comparisons must never turn bad evidence into a pass."""
    reference, args = reference_inputs(96)
    reference["gpu4pyscf"][key] = value
    with pytest.raises(RuntimeError, match=message):
        independent_reference(reference, *args)
