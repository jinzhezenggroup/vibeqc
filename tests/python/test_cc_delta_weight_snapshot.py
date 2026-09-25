"""Independent delta-Lambda checks must not share reusable executor storage."""

from types import SimpleNamespace

import numpy as np
import pytest

from tools.vibeqc_cc import triples_lambda_response as response_module


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("mode", ["mismatch", "equal", "near"])
def test_delta_weight_snapshots_each_execution(
    monkeypatch: pytest.MonkeyPatch, external: bool, mode: str
) -> None:
    buffer = np.empty((2, 2), dtype=np.float64)
    first = np.array([[0.125, -0.5], [0.75, 1.0]])
    shift = {"mismatch": 0.25, "equal": 0.0, "near": 5e-13}[mode]
    calls: list[str] = []

    def execute(program: str, feeds: object) -> dict[str, np.ndarray]:
        calls.append(program)
        buffer[:] = first + (shift if len(calls) == 2 else 0.0)
        return {"bar_fov": buffer}

    monkeypatch.setattr(
        response_module,
        "build_parameter_vjp",
        lambda primal, parameter: SimpleNamespace(program=primal),
    )
    bound = SimpleNamespace(
        programs=SimpleNamespace(primal="shared"),
        independent=SimpleNamespace(primal="independent"),
        feeds={},
        _tensor_execute=execute,
    )
    # Isolate ownership of the two graph results, not Lambda construction or
    # physical equations; those retain their independent endpoint test suites.
    owner = object.__new__(response_module.BoundCCSDTResponse)
    object.__setattr__(owner, "bound", bound)
    object.__setattr__(
        owner,
        "corrected",
        SimpleNamespace(
            delta_lambda1=np.zeros((1, 1)),
            delta_lambda2=np.zeros((1, 1, 1, 1)),
        ),
    )
    object.__setattr__(
        owner,
        "parameter_executor",
        SimpleNamespace(execute=execute) if external else None,
    )
    if mode == "mismatch":
        with pytest.raises(
            response_module.ImplicitSolveError, match="independent corrected-Lambda"
        ):
            owner._delta_weight("fov")
    else:
        result, error = owner._delta_weight("fov")
        np.testing.assert_array_equal(result, first)
        assert error == np.max(np.abs(first - (first + shift)))
        buffer[:] = 99.0
        np.testing.assert_array_equal(result, first)
        assert not np.shares_memory(result, buffer)
    assert calls == ["shared", "independent"]
