"""Schedule admission may not publish a different scientific workload identity."""

from dataclasses import replace

import pytest
from test_schedule_contract import _tensor_contract
from vibeqc_compiler.dft.xc_schedule import (
    DEVICE_FUSED,
    GridXcCandidateLimits,
    GridXcCandidateShape,
    GridXcScientificIdentity,
    assess_grid_xc_schedule,
)


@pytest.mark.parametrize(
    "changes",
    ({"functional": "R2SCAN"}, {"observable": "energy"}, {"spin": "unpolarized"}),
)
def test_dft_identity_must_match_admitted_workload(changes: dict) -> None:
    scientific = GridXcScientificIdentity(
        "sm_120",
        "PBE",
        "f" * 64,
        ("rho", "gradient", "sigma"),
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
        "g" * 64,
        "v2",
        None,
        "fp64",
        "polarized",
        "potential",
        "density_matrix",
        "s" * 64,
    )
    with pytest.raises(ValueError, match="scientific identity"):
        assess_grid_xc_schedule(
            DEVICE_FUSED,
            GridXcCandidateShape(64, 16, 4, 4, 2, 4, 1024, 512),
            GridXcCandidateLimits(8192, 100000, 8192),
            device_xc_available=True,
            observable="potential",
            functional="PBE",
            scientific=replace(scientific, **changes),
        )


def test_schedule_identity_is_not_optional() -> None:
    with pytest.raises(ValueError, match="schedule_hash"):
        replace(_tensor_contract(), schedule_hash=None)
