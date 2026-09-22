"""Changed endpoints include reset work and reject nonconverged oracles."""

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import issue206_rebuild as runner


def _engine(clock: list[float], events: list[str], converged: bool) -> SimpleNamespace:
    def reset(molecule: object) -> None:
        clock[0] += 7.0
        events.append("reset")

    def kernel(*, dm0: object) -> float:
        clock[0] += 3.0
        events.append("kernel")
        return -1.0

    def gradient() -> np.ndarray:
        clock[0] += 2.0
        events.append("gradient")
        return np.zeros((2, 3))

    return SimpleNamespace(
        mol=SimpleNamespace(set_geom_=lambda *args, **kwargs: object()),
        reset=reset,
        kernel=kernel,
        converged=converged,
        nuc_grad_method=lambda: SimpleNamespace(kernel=gradient),
    )


def test_changed_reference_timer_includes_geometry_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock, events = [0.0], []
    monkeypatch.setattr(runner, "_sync", lambda cp: events.append("sync"))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(runner, "gpu_convergence_payload", lambda *args: [])
    engine = _engine(clock, events, True)
    seed = np.eye(2)
    result = runner._stock_sample(
        [engine],
        [seed],
        SimpleNamespace(asnumpy=np.asarray),
        systems=[[("H", (0, 0, 0)), ("H", (0, 0, 1))]],
        coordinates=[np.zeros((2, 3))],
    )
    assert result["seconds"] == 12.0
    assert events == ["sync", "reset", "kernel", "gradient", "sync"]
    np.testing.assert_array_equal(seed, np.eye(2))


def test_nonconverged_reference_is_rejected_before_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    monkeypatch.setattr(runner, "_sync", lambda cp: None)
    monkeypatch.setattr(runner, "gpu_convergence_payload", lambda *args: [])
    engine = _engine([0.0], events, False)
    with pytest.raises(RuntimeError, match="converg"):
        runner._stock_sample([engine], [np.eye(2)], SimpleNamespace(asnumpy=np.asarray))
    assert "gradient" not in events


def test_streamed_route_does_not_satisfy_explicit_fallback() -> None:
    from benchmarks.issue206_resident_sentinel import validate_response_record

    record = {
        "operation": "force_response",
        "nbf": 4,
        "naux": 5,
        "source_backed": True,
        "counters": {},
        "tiles": [],
    }
    with pytest.raises(RuntimeError, match="fallback"):
        validate_response_record(record, expected_policy="fallback")
    assert (
        validate_response_record(record, expected_policy="streamed")["policy"][
            "residency"
        ]
        == "streamed"
    )


@pytest.mark.parametrize(
    "upload,scatter", [("drain", ""), ("packed", ""), ("", "sharded")]
)
def test_host_fallback_rejects_unexecuted_attribution_probes(
    upload: str, scatter: str
) -> None:
    from benchmarks import issue308_response_timeline as timeline

    with pytest.raises(RuntimeError, match="fallback"):
        timeline.validate_host_fallback_probes(upload, scatter)
    timeline.validate_host_fallback_probes("", "")


@pytest.mark.parametrize(
    "malformed", ["batch", "atoms", "coordinates", "scalar", "complex", "nan"]
)
def test_endpoint_comparison_rejects_incomplete_or_nonphysical_arrays(
    malformed: str,
) -> None:
    reference = {
        "energies_hartree": np.array([-1.0, -1.0]),
        "forces_hartree_per_bohr": np.zeros((2, 2, 3)),
    }
    candidate = {key: value.copy() for key, value in reference.items()}
    if malformed == "batch":
        candidate = {key: value[:1] for key, value in candidate.items()}
    elif malformed == "atoms":
        candidate["forces_hartree_per_bohr"] = np.zeros((2, 1, 3))
    elif malformed == "coordinates":
        candidate["forces_hartree_per_bohr"] = np.zeros((2, 2, 1))
    elif malformed == "scalar":
        candidate["energies_hartree"] = np.array(-1.0)
    elif malformed == "complex":
        candidate["energies_hartree"] = candidate["energies_hartree"].astype(complex)
    else:
        candidate["forces_hartree_per_bohr"][0, 0, 0] = np.nan
    with pytest.raises((ValueError, TypeError), match="endpoint"):
        runner._paired_errors(candidate, reference)


def test_endpoint_comparison_preserves_complete_numerical_errors() -> None:
    reference = {
        "energies_hartree": [-1.0, -2.0],
        "forces_hartree_per_bohr": np.zeros((2, 2, 3)),
    }
    candidate = {
        "energies_hartree": [-1.25, -2.0],
        "forces_hartree_per_bohr": np.zeros((2, 2, 3)),
    }
    candidate["forces_hartree_per_bohr"][1, 1, 2] = -0.5
    assert runner._paired_errors(candidate, reference) == {
        "maximum_energy_error_hartree": 0.25,
        "maximum_force_error_hartree_per_bohr": 0.5,
    }
