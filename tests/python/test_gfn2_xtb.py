import os

import numpy as np
import pytest
from vibeqc import Calculator, method_capabilities

H3_PLUS = [
    ("H", (-0.47073898552969, 0.81534384004086, 0.0)),
    ("H", (-0.47073898552969, -0.81534384004086, 0.0)),
    ("H", (0.94147797105939, 0.0, 0.0)),
]
H3_PLUS_ENERGY = -0.8989438125591571
H3_PLUS_FORCES = np.array(
    [
        [-0.04009016325520178, 0.06943819964174042, -0.0],
        [-0.04009016325520183, -0.06943819964174044, -0.0],
        [0.08018032651040363, 2.986512564642647e-17, -0.0],
    ]
)

OH_RADICAL = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.834))]
OH_ENERGY = -4.42833345932
OH_FORCES = np.array(
    [
        [6.3926072946487e-20, -7.153190104741e-18, 0.006059584199658],
        [-6.3926072946487e-20, 7.153190104741e-18, -0.006059584199658],
    ]
)


def test_gfn2_capability_and_intrinsic_basis_contract() -> None:
    capability = method_capabilities("gfn2-xtb")
    assert capability.family == "semiempirical"
    assert capability.available
    assert not capability.supports_batch
    assert capability.supported_properties == frozenset(("energy", "forces"))

    calculator = Calculator(method="gfn2-xtb", device="cpu")
    metadata = calculator.basis_metadata(H3_PLUS, charge=1)
    assert metadata["intrinsic_basis"]["name"] == "GFN2-xTB intrinsic minimal basis"
    assert metadata["intrinsic_basis"]["element_domain"] == [1, 86]

    with pytest.raises(ValueError, match="intrinsic minimal basis"):
        Calculator(method="gfn2-xtb", basis="sto-3g", device="cpu")


def test_gfn2_h3_plus_matches_independent_tblite_golden() -> None:
    """Gate charge, total energy and analytic forces against tblite 0.7.0.

    Golden provenance is copied numerically from xTBloom's independently
    generated data/conformance/golden/h3_plus.json (tblite revision
    e9abc395b122018ed688aecb1c3a65cecaf97beb).
    """

    result = Calculator(method="gfn2-xtb", device="cpu").singlepoint(H3_PLUS, charge=1)
    assert result.converged
    assert result.energy == pytest.approx(H3_PLUS_ENERGY, abs=5.0e-7)
    assert np.allclose(result.forces, H3_PLUS_FORCES, atol=5.0e-7, rtol=0.0)
    assert np.max(np.abs(result.forces.sum(axis=0))) < 1.0e-12
    assert result.executed_backend == "cpu_reference"


def test_gfn2_oh_radical_matches_independent_xtb_golden() -> None:
    """Gate standard restricted open-shell GFN2 against xTB 6.7.1.

    Golden provenance is copied numerically from xTBloom's independently
    generated data/conformance/golden/oh_radical.json (xTB revision
    edcfbbe39d411edc225e27315fbda3a204ddb023).
    """

    calculator = Calculator(method="gfn2-xtb", device="cpu")
    result = calculator.singlepoint(OH_RADICAL, multiplicity=2)
    assert result.converged
    assert result.energy == pytest.approx(OH_ENERGY, abs=5.0e-7)
    assert np.allclose(result.forces, OH_FORCES, atol=5.0e-7, rtol=0.0)

    with pytest.raises(RuntimeError, match="integral spin occupations"):
        calculator.singlepoint(OH_RADICAL)


def test_gfn2_reports_scc_nonconvergence_explicitly() -> None:
    with pytest.raises(RuntimeError, match="SCF did not converge"):
        Calculator(method="gfn2-xtb", device="cpu", max_iterations=1).singlepoint(
            H3_PLUS, charge=1, properties=("energy",)
        )


def test_gfn2_oh_analytic_force_matches_energy_finite_difference() -> None:
    calculator = Calculator(
        method="gfn2-xtb",
        device="cpu",
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
    )
    z = 1.834

    def energy(displacement: float) -> float:
        return calculator.singlepoint(
            [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, z + displacement))],
            multiplicity=2,
            properties=("energy",),
        ).energy

    result = calculator.singlepoint(
        OH_RADICAL, multiplicity=2, properties=("energy", "forces")
    )
    step = 1.0e-4
    finite_difference_force = -(energy(step) - energy(-step)) / (2.0 * step)
    assert result.forces[1, 2] == pytest.approx(finite_difference_force, abs=2.0e-8)


def _cuda_gfn2_singlepoint_or_skip(
    atoms: list[tuple[str, tuple[float, float, float]]],
    *,
    charge: int = 0,
    multiplicity: int = 1,
) -> object:
    if os.environ.get("VIBEQC_TEST_GFN2_CUDA") != "1":
        pytest.skip("set VIBEQC_TEST_GFN2_CUDA=1 in a qualified GPU allocation")
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("GFN2 CUDA qualification requires a Slurm allocation")
    calculator = Calculator(
        method="gfn2-xtb",
        device="cuda",
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
    )

    # Explicit device qualification must fail, not skip, when the requested
    # runtime capability is absent or its numerical execution fails.
    return calculator.singlepoint(atoms, charge=charge, multiplicity=multiplicity)


def test_gfn2_cuda_h3_plus_matches_independent_tblite_golden() -> None:
    result = _cuda_gfn2_singlepoint_or_skip(H3_PLUS, charge=1)
    assert result.converged
    assert result.executed_backend == "cuda"
    assert result.energy == pytest.approx(H3_PLUS_ENERGY, abs=5.0e-7)
    assert np.allclose(result.forces, H3_PLUS_FORCES, atol=5.0e-7, rtol=0.0)
    assert np.max(np.abs(result.forces.sum(axis=0))) < 1.0e-12


def test_gfn2_cuda_oh_radical_matches_independent_xtb_golden() -> None:
    result = _cuda_gfn2_singlepoint_or_skip(OH_RADICAL, multiplicity=2)
    assert result.converged
    assert result.executed_backend == "cuda"
    assert result.energy == pytest.approx(OH_ENERGY, abs=5.0e-7)
    assert np.allclose(result.forces, OH_FORCES, atol=5.0e-7, rtol=0.0)


def test_explicit_cuda_qualification_does_not_skip_missing_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from typing import Any

    class Unavailable:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def singlepoint(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError("library was built without CUDA support")

    monkeypatch.setenv("VIBEQC_TEST_GFN2_CUDA", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "host-admission-fixture")
    monkeypatch.setattr(sys.modules[__name__], "Calculator", Unavailable)
    with pytest.raises(NotImplementedError, match="without CUDA support"):
        _cuda_gfn2_singlepoint_or_skip([("H", (0.0, 0.0, 0.0))])
