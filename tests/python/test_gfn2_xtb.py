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
    with pytest.raises(NotImplementedError, match="CUDA"):
        Calculator(method="gfn2-xtb", device="cuda")


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
