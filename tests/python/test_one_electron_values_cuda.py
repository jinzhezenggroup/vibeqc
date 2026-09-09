"""Fail-closed real-GPU endpoint gates for one-electron value candidates.

Run with VIBEQC_ONE_ELECTRON_CUDA_TEST=1 under Slurm. Once enabled, CUDA
runtime errors fail these tests; they cannot become unavailable-hardware skips.
"""

import os

import numpy as np
import pytest
from vibeqc import Calculator, Primitive, Shell

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_ONE_ELECTRON_CUDA_TEST") != "1",
    reason="explicit Slurm GPU numerical tier",
)


def run_case(monkeypatch, selection, mapping, *, method, representation, fitted, count):
    """Exercise cold, unchanged and changed geometry on one fixed topology."""
    monkeypatch.setenv("VIBEQC_ONE_ELECTRON_VALUES", selection)
    monkeypatch.setenv("VIBEQC_ONE_ELECTRON_VALUE_MAPPING", mapping)
    # An s/d/f atom plus a separate s atom covers sparse real-spherical f
    # expansions without making this one-electron gate a large ERI benchmark.
    basis = (
        Shell(0, 0, (Primitive(1.5, 1.0), Primitive(0.7, -0.1))),
        Shell(0, 2, (Primitive(0.8, 1.0),)),
        Shell(0, 3, (Primitive(0.6, 1.0),)),
        Shell(1, 0, (Primitive(1.2, 1.0),)),
    )
    systems = [
        [("He", (0.0, 0.0, -0.7 - 0.1 * i)), ("H", (0.1, 0.0, 0.7))]
        for i in range(count)
    ]
    charge, multiplicity = (1, 1) if method == "rhf" else (0, 2)
    calculator = Calculator(
        method=method,
        device="cuda",
        basis=basis,
        basis_representation=representation,
        density_fitting="cuda" if fitted else "none",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
    )
    with calculator.prepare_batch(
        systems, charges=[charge] * count, multiplicities=[multiplicity] * count
    ) as prepared:
        cold = prepared.execute(strict=True)
        warm = prepared.execute(strict=True)
        moved_positions = [
            np.array([position for _, position in system]) for system in systems
        ]
        for i, positions in enumerate(moved_positions):
            positions[1, 0] += 0.013 * (i + 1)
        moved = prepared.execute(moved_positions, strict=True)
        # Returning to the original geometry must recompute from its own atoms.
        restored = prepared.execute(
            [np.array([r for _, r in system]) for system in systems], strict=True
        )
    return cold, warm, moved, restored


@pytest.mark.parametrize("method", ["rhf", "uhf"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("count", [1, 3])
def test_generated_schedules_preserve_scf_and_geometry(
    monkeypatch, method, representation, fitted, count
):
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run inside Slurm"
    kwargs = {
        "method": method,
        "representation": representation,
        "fitted": fitted,
        "count": count,
    }
    reference = run_case(monkeypatch, "reference", "thread", **kwargs)
    for mapping in ("thread", "shell_warp"):
        actual = run_case(monkeypatch, "generated", mapping, **kwargs)
        for expected, found in zip(reference, actual):
            np.testing.assert_allclose(
                found.energies, expected.energies, atol=3e-10, rtol=0
            )
            for a, b in zip(found.items, expected.items):
                assert a.executed_backend == "cuda"
                np.testing.assert_allclose(a.forces, b.forces, atol=3e-9, rtol=0)
        np.testing.assert_allclose(
            actual[0].energies, actual[1].energies, atol=3e-10, rtol=0
        )
        np.testing.assert_allclose(
            actual[0].energies, actual[3].energies, atol=3e-10, rtol=0
        )
        assert np.max(np.abs(actual[0].energies - actual[2].energies)) > 1e-7


def test_policy_changes_rebuild_reused_direct_plan(monkeypatch):
    """A cached plan must follow the policy recorded for the current execution."""
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run inside Slurm"
    atoms = [("H", (0, 0, -0.7)), ("H", (0.1, 0, 0.7))]
    moved = np.array([r for _, r in atoms], dtype=float)
    moved[1, 0] += 0.01
    calc = Calculator(device="cuda", energy_tolerance=1e-12, density_tolerance=1e-10)
    monkeypatch.setenv("VIBEQC_ONE_ELECTRON_VALUES", "reference")
    monkeypatch.setenv("VIBEQC_ONE_ELECTRON_VALUE_MAPPING", "thread")
    with calc.prepare_batch([atoms]) as prepared:
        prepared.execute(strict=True)
        for selection, mapping in (
            ("generated", "thread"),
            ("generated", "shell_warp"),
            ("reference", "thread"),
        ):
            monkeypatch.setenv("VIBEQC_ONE_ELECTRON_VALUES", selection)
            monkeypatch.setenv("VIBEQC_ONE_ELECTRON_VALUE_MAPPING", mapping)
            actual = prepared.execute([moved], strict=True)
            with calc.prepare_batch([[("H", tuple(r)) for r in moved]]) as fresh:
                expected = fresh.execute(strict=True)
            np.testing.assert_allclose(
                actual.energies, expected.energies, atol=3e-10, rtol=0
            )
            np.testing.assert_allclose(
                actual.items[0].forces, expected.items[0].forces, atol=3e-9, rtol=0
            )
            controls = prepared._warm_metadata[0]["controls"]["runtime_policy"]
            assert controls["VIBEQC_ONE_ELECTRON_VALUES"] == selection
            assert controls["VIBEQC_ONE_ELECTRON_VALUE_MAPPING"] == mapping
