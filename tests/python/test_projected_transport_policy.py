"""A validated projection policy must not borrow a mutable tolerance container."""

import numpy as np
import pytest
from test_projected_state_transport import _projected_transport, _source_amplitudes

from tools.vibeqc_cc.projected_transport import (
    ProjectedAmplitudePolicy,
    project_amplitude_guess,
)


def test_caller_mutation_cannot_widen_projection_gate() -> None:
    tolerance = np.array(1e-8)
    policy = ProjectedAmplitudePolicy(map_tolerance=tolerance)
    tolerance[...] = 0.1
    assert type(policy.map_tolerance) is float
    assert policy.map_tolerance == 1e-8
    with pytest.raises(ValueError, match="bounded overlap contraction"):
        project_amplitude_guess(
            _projected_transport(virtual_scale=1.01),
            *_source_amplitudes(),
            policy=policy,
        )


@pytest.mark.parametrize("value", [1e-8, np.float64(1e-8), np.array(1e-8)])
def test_real_scalar_policy_keeps_existing_projection(value: object) -> None:
    policy = ProjectedAmplitudePolicy(map_tolerance=value)
    assert type(policy.map_tolerance) is float
    guess = project_amplitude_guess(
        _projected_transport(), *_source_amplitudes(), policy=policy
    )
    np.testing.assert_array_equal(guess.amplitudes.t1, [[0.2, 0.0]])
    assert guess.kind == "projected_warm_start"


@pytest.mark.parametrize(
    "value",
    [
        np.array([1e-8]),
        np.array(1e-8 + 1j),
        "1e-8",
        None,
        float("nan"),
        float("inf"),
        0.0,
        -1e-8,
        1e-3,
    ],
)
def test_invalid_tolerance_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="map_tolerance"):
        ProjectedAmplitudePolicy(map_tolerance=value)
