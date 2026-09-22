"""A malformed timing record must never become successful benchmark evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

from tools.benchmark_stationary_cuda_timeline import _successful_record


@pytest.mark.parametrize(
    "case",
    [
        "external_nan",
        "external_inf",
        "export_nan",
        "export_negative",
        "phase_nan",
        "energy_nan",
        "gradient_layout",
    ],
)
def test_success_record_rejects_invalid_scientific_or_timing_data(case: str) -> None:
    timeline = {
        "exclusive_wall_seconds": {"execute": 1.0},
        "endpoint_seconds": 1.0,
        "measurement_policy": "test",
    }
    work = {"timeline": timeline, "endpoint_seconds": 1.0}
    result = SimpleNamespace(work=work, gradient=np.zeros((1, 3)))
    kwargs = {
        "system": "test",
        "method": "pbe-rks",
        "scenario": "cold",
        "repeat": 0,
        "state_export_seconds": 0.0,
        "observed_diagnostic_seconds": 1.0,
        "result": result,
        "energy": -1.0,
        "basis": SimpleNamespace(natom=1, nao=1, nprimitive=1),
    }
    if case == "external_nan":
        kwargs["observed_diagnostic_seconds"] = float("nan")
    elif case == "external_inf":
        kwargs["observed_diagnostic_seconds"] = float("inf")
    elif case == "export_nan":
        kwargs["state_export_seconds"] = float("nan")
    elif case == "export_negative":
        kwargs["state_export_seconds"] = -0.25
    elif case == "phase_nan":
        timeline["exclusive_wall_seconds"]["execute"] = float("nan")
    elif case == "energy_nan":
        kwargs["energy"] = float("nan")
    else:
        result.gradient = np.zeros(3)
    with pytest.raises((ValueError, RuntimeError)):
        _successful_record(**kwargs)


def test_finite_valid_record_still_reconciles() -> None:
    work = {
        "endpoint_seconds": 1.0,
        "timeline": {
            "exclusive_wall_seconds": {"execute": 1.0},
            "endpoint_seconds": 1.0,
            "measurement_policy": "test",
        },
    }
    record = _successful_record(
        system="test",
        method="pbe-rks",
        scenario="cold",
        repeat=0,
        state_export_seconds=0.2,
        observed_diagnostic_seconds=1.0,
        result=SimpleNamespace(work=work, gradient=np.zeros((1, 3))),
        energy=-1.0,
        basis=SimpleNamespace(natom=1, nao=1, nprimitive=1),
    )
    assert record["status"] == "ok"
    assert record["timeline"]["reconciliation_error_seconds"] == 0.0
