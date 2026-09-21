"""Complete trajectory timing includes preparation, failed retries and teardown."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from test_force_aware_scf import policy

from tools.vibeqc_numerics import scf_effort_geomopt_benchmark as bench

if TYPE_CHECKING:
    import pytest
    from typing_extensions import Self
    from vibeqc.force_aware_scf import ScfEffortLevel


def test_complete_policy_time_includes_failed_solve_and_owner_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    controller = policy(relax_after=1)
    count = [0]

    def endpoint(force: float) -> SimpleNamespace:
        return SimpleNamespace(
            forces=((force, 0.0, 0.0), (-force, 0.0, 0.0)),
            seconds=2.0,
            iterations=3,
            converged=True,
            energy=-1.0,
            energy_change=1e-12,
            density_rms=1e-10,
        )

    class Prepared:
        def __init__(self, *args: object) -> None:
            pass

        def __enter__(self) -> Self:
            clock[0] += 3.0
            return self

        def __exit__(self, *args: object) -> None:
            clock[0] += 4.0

        def evaluate(
            self, coordinates: object, selected: ScfEffortLevel
        ) -> SimpleNamespace:
            count[0] += 1
            if count[0] == 2:
                assert not selected.strict
                clock[0] += 7.0
                raise RuntimeError("failed loose solve")
            clock[0] += 2.0
            return endpoint(0.1 if count[0] == 1 else 0.0)

    def cleanup(*args: object, **kwargs: object) -> SimpleNamespace:
        clock[0] += 5.0
        return endpoint(0.0)

    monkeypatch.setattr(bench.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(bench, "PreparedLevels", Prepared)
    monkeypatch.setattr(bench, "single_endpoint", cleanup)
    monkeypatch.setattr(bench, "basis_calibration_id", lambda case: "sto-3g")
    monkeypatch.setattr(bench, "scientific_model_id", lambda *args: "target")
    case = bench.Case("pair", "pair", (("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))))
    result = bench.fire_optimize(
        case,
        device="cpu",
        policy=controller,
        adaptive=True,
        precision="fp64",
        max_steps=2,
    )
    assert result["retries"] == 1
    assert result["complete_policy_seconds"] == 23.0
    assert result["production_seconds_before_cleanup"] == 18.0
    assert result["final_verification"]["status"] == "observed_met"
