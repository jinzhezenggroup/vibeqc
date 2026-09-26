"""Scientific reference admission must fail before permissive NumPy arithmetic."""

import json
import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.df_policy_endpoint import (
    CASES,
    endpoint_errors,
    independent_reference,
    require_frozen_checkpoint_geometry,
)

EVIDENCE = (
    Path(__file__).resolve().parents[2] / "benchmarks/results/issue377-379-df/gpu4pyscf"
)

# This is the explicit retained-oracle inventory, not every executable workload.
# New qualification-only endpoints need fresh independent reference generation.
# Never discover this list from existing files: losing a retained fixture must
# still fail the gate instead of silently reducing coverage.
RETAINED_REFERENCE_AOS = (96, 192, 384, 768)


def test_frozen_benchmark_checkpoint_rejects_changed_geometry() -> None:
    manifest = SimpleNamespace(
        items=({"model": {"geometry_hash": "source"}, "controls": "different"},)
    )
    require_frozen_checkpoint_geometry(manifest, "source")
    with pytest.raises(RuntimeError, match="checkpoint geometry differs"):
        require_frozen_checkpoint_geometry(manifest, "target")


def reference_inputs(aos: typing.Any) -> typing.Any:
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


@pytest.mark.parametrize("aos", RETAINED_REFERENCE_AOS)
def test_retained_independent_references_match_current_workloads(
    aos: typing.Any,
) -> None:
    """All four retained geometries and scientific settings remain admissible."""
    reference, args = reference_inputs(aos)
    energy, forces = independent_reference(reference, *args)
    np.testing.assert_array_equal(energy, args[1])
    np.testing.assert_array_equal(forces, args[2])


@pytest.mark.parametrize("aos", (648, 864))
def test_qualification_workloads_reject_unrelated_retained_reference(aos: int) -> None:
    """New inputs cannot inherit scientific qualification from a smaller case."""
    assert aos in CASES and aos not in RETAINED_REFERENCE_AOS
    reference, args = reference_inputs(96)
    with pytest.raises(RuntimeError, match="workload differs: case"):
        independent_reference(reference, aos, *args[1:])


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
def test_reference_rejects_changed_workload(key: typing.Any, value: typing.Any) -> None:
    """A matching case label cannot authorize a different scientific workload."""
    reference, args = reference_inputs(96)
    reference["workload"][key] = value
    with pytest.raises(RuntimeError, match=f"workload differs: {key}"):
        independent_reference(reference, *args)


@pytest.mark.parametrize("change", ["coordinate", "element", "basis"])
def test_reference_rejects_changed_geometry_or_basis(change: typing.Any) -> None:
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
def test_reference_rejects_broadcastable_or_nonfinite_arrays(
    key: typing.Any, value: typing.Any, message: typing.Any
) -> None:
    """Broadcasting and NaN comparisons must never turn bad evidence into a pass."""
    reference, args = reference_inputs(96)
    reference["gpu4pyscf"][key] = value
    with pytest.raises(RuntimeError, match=message):
        independent_reference(reference, *args)


@pytest.mark.parametrize(
    "controls",
    [
        {"dense": {"VIBEQC_DF_FINAL_EXCHANGE": "dense"}},
        {"dense": {"CUDA_VISIBLE_DEVICES": "0"}, "auto": {"CUDA_VISIBLE_DEVICES": "0"}},
        {"unknown": {}},
    ],
)
def test_coupled_policies_reject_incomplete_or_non_schedule_arms(
    controls: typing.Any,
    monkeypatch: typing.Any,
    tmp_path: typing.Any,
    capsys: typing.Any,
) -> None:
    """Reject settings that could leak between arms or change scheduler visibility."""
    from benchmarks.df_policy_endpoint import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "df_policy_endpoint",
            "--aos",
            "768",
            "--output",
            str(tmp_path / "run.json"),
            "--control",
            "VIBEQC_DF_SEED_EXCHANGE",
            "--policies",
            "dense",
            "auto",
            "--policy-controls",
            json.dumps(controls),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "policy-controls" in capsys.readouterr().err


@pytest.mark.parametrize("side", ("actual", "reference"))
@pytest.mark.parametrize("bad", ("shape", "nan", "inf"))
def test_cpu_reference_endpoint_checks_reject_bad_arrays(
    side: typing.Any, bad: typing.Any
) -> None:
    arrays = [np.zeros(1), np.zeros((1, 2, 3)), np.zeros(1), np.zeros((1, 2, 3))]
    index = 1 if side == "actual" else 3
    if bad == "shape":
        arrays[index] = np.zeros((1, 1, 3))
    else:
        arrays[index][0, 0, 0] = float(bad)
    with pytest.raises(RuntimeError, match="arrays are invalid"):
        endpoint_errors(*arrays)


def test_cpu_reference_endpoint_energy_only_and_force_errors() -> None:
    assert endpoint_errors([1.0], None, [1.25], None) == (0.25, None)
    assert endpoint_errors([1.0], [[[1.0, 2.0, 3.0]]], [1.25], [[[1.0, 1.5, 3.0]]]) == (
        0.25,
        0.5,
    )
