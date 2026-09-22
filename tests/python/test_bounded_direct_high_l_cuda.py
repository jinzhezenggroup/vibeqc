"""Bounded Direct-HF must retain exact high-angular Fock and force coverage."""

import os

import numpy as np
import pytest
from vibeqc import Calculator, Primitive, Shell

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)

ATOMS = (("He", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7)))
BASIS = (
    Shell(0, 0, (Primitive(1.5, 1.0),)),
    Shell(0, 2, (Primitive(0.8, 1.0),)),
    Shell(0, 3, (Primitive(0.6, 1.0),)),
    Shell(1, 0, (Primitive(1.2, 1.0),)),
)


@pytest.mark.parametrize(
    ("method", "charge", "multiplicity"),
    (("rhf", 1, 1), ("uhf", 0, 2)),
)
def test_forced_bounded_high_l_matches_fixed(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    charge: int,
    multiplicity: int,
) -> None:
    """Registry gaps use the bounded high-l oracle instead of failing late."""

    assert os.environ.get("SLURM_JOB_ID")
    monkeypatch.delenv("VIBEQC_BOUNDED_DIRECT_STREAMING", raising=False)
    monkeypatch.delenv("VIBEQC_DIRECT_TILE_VALIDATION", raising=False)
    options = {
        "method": method,
        "basis": BASIS,
        "device": "cuda",
        "density_fitting": "none",
        "energy_tolerance": 1.0e-12,
        "density_tolerance": 1.0e-10,
        "screening_tolerance": 1.0e-14,
    }
    try:
        fixed = Calculator(**options).singlepoint(
            ATOMS, charge=charge, multiplicity=multiplicity
        )
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable before bounded override: {error}")

    monkeypatch.setenv("VIBEQC_BOUNDED_DIRECT_STREAMING", "force")
    bounded = Calculator(**options).singlepoint(
        ATOMS, charge=charge, multiplicity=multiplicity
    )

    assert fixed.executed_backend == bounded.executed_backend == "cuda"
    assert bounded.energy == pytest.approx(fixed.energy, abs=2.0e-10)
    np.testing.assert_allclose(bounded.forces, fixed.forces, atol=6.0e-9, rtol=0)
