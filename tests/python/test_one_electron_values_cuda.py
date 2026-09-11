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


def sdf_case_inputs(method, count):
    """Share physical fixtures between replay tests and independent oracles."""
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
    return basis, systems, charge, multiplicity


def run_case(
    monkeypatch, mapping, *, method, representation, fitted, count, device="cuda"
):
    """Exercise cold, unchanged and changed geometry on one fixed topology."""
    if mapping is None:
        monkeypatch.delenv("VIBEQC_ONE_ELECTRON_VALUE_MAPPING", raising=False)
    else:
        monkeypatch.setenv("VIBEQC_ONE_ELECTRON_VALUE_MAPPING", mapping)
    basis, systems, charge, multiplicity = sdf_case_inputs(method, count)
    calculator = Calculator(
        method=method,
        device=device,
        basis=basis,
        basis_representation=representation,
        density_fitting=device if fitted else "none",
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
    # Same-expression schedule parity is distinct from independent correctness:
    # Libcint below and archived clean-baseline endpoint evidence own that gate.
    reference = run_case(monkeypatch, "thread", **kwargs)
    for mapping in ("shell_warp", None):
        actual = run_case(monkeypatch, mapping, **kwargs)
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
    monkeypatch.setenv("VIBEQC_ONE_ELECTRON_VALUE_MAPPING", "thread")
    with calc.prepare_batch([atoms]) as prepared:
        prepared.execute(strict=True)
        for mapping in ("thread", "shell_warp", "thread"):
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
            assert "VIBEQC_ONE_ELECTRON_VALUES" not in controls
            assert "VIBEQC_DF_VALUES" not in controls
            assert controls["VIBEQC_ONE_ELECTRON_VALUE_MAPPING"] == mapping


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("mapping", ["thread", "shell_warp"])
def test_generated_pair_policy_hcore_matches_independent_libcint(
    monkeypatch, representation, mapping
):
    """Exercise normalized pair traversal independently of an SCF fixed point.

    Negative contraction coefficients, every s/p/d/f shell, unequal charges
    and moved nuclei cover the policy's weighting, component and nuclear loops.
    The independent oracle applies its own Cartesian normalization convention.
    """
    from vibeqc.fock import FockBuildSpec, FockPlan, FockTerm
    from vibeqc_compiler.dft import NativeAO
    from vibeqc_compiler.dft.fixtures import basis_arguments

    from tools.generate_validation_references import pyscf_molecule

    assert os.environ.get("SLURM_JOB_ID"), "GPU tests require Slurm"
    monkeypatch.setenv("VIBEQC_ONE_ELECTRON_VALUE_MAPPING", mapping)
    inputs = {
        "atomic_numbers": [2, 1],
        "coordinates": [[0.2, -0.3, 0.1], [-0.4, 0.15, 0.8]],
        "shells": [
            {
                "atom_index": atom,
                "angular_momentum": angular,
                "primitives": [[0.35 + 0.4 * atom, 1.0], [1.13, -0.15]],
            }
            for atom in (0, 1)
            for angular in range(4)
        ],
        "basis_representation": representation,
        "charge": 1,
        "multiplicity": 1,
    }
    # No ERI tensor is needed to validate the one-electron production consumer.
    spec = FockBuildSpec(
        coulomb=FockTerm(False), exchange=FockTerm(False), derivative_order=0
    )
    for displacement in (0.0, 0.017):
        inputs["coordinates"][1][0] += displacement
        mol, scale, _ = pyscf_molecule(inputs)
        expected = (mol.intor("int1e_kin") + mol.intor("int1e_nuc")) * (
            scale[:, None] * scale[None, :]
        )
        with (
            NativeAO(**basis_arguments({"inputs": inputs})) as basis,
            FockPlan(basis, spec, device="cuda") as plan,
        ):
            actual = plan.evaluate(np.eye(expected.shape[0])).fock
            np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=3e-12)
            if representation == "cartesian":
                np.testing.assert_array_equal(actual, actual.T)
            else:
                # This public source transforms Cartesian matrices with two
                # library contractions; their opposite reduction orders may
                # differ by roundoff despite exact symmetry at the pair store.
                np.testing.assert_allclose(actual, actual.T, atol=1e-14, rtol=0)
