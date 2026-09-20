"""Reject malformed host evidence before a timeline sample is marked successful."""

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.stationary_cuda_timeline import _record


@pytest.mark.parametrize(
    "case",
    [
        "endpoint_nan",
        "endpoint_inf",
        "endpoint_negative",
        "export_nan",
        "phase_nan",
        "gradient_nan",
        "gradient_shape",
    ],
)
def test_timeline_rejects_invalid_sample(case: str) -> None:
    work = dict.fromkeys(
        (
            "launches",
            "primitive_records",
            "xc_points",
            "grid_pair_visits",
            "tensor_executions",
            "tensor_work",
            "snapshot_export_work",
            "additional_device_peak_bound",
            "additional_host_numeric_bound",
            "endpoint_seconds",
        ),
        0,
    )
    work["timeline"] = {"exclusive_seconds": {"compute": 1.0}}
    result = SimpleNamespace(work=work, gradient=np.zeros((2, 3)))
    endpoint, exported = 1.0, 0.0
    if case.startswith("endpoint_"):
        endpoint = {
            "endpoint_nan": float("nan"),
            "endpoint_inf": float("inf"),
            "endpoint_negative": -1.0,
        }[case]
    elif case == "export_nan":
        exported = float("nan")
    elif case == "phase_nan":
        work["timeline"]["exclusive_seconds"]["compute"] = float("nan")
    elif case == "gradient_nan":
        result.gradient[0, 0] = np.nan
    else:
        result.gradient = np.zeros((3, 2))
    with pytest.raises(ValueError, match="finite|gradient"):
        _record("cold", endpoint, exported, SimpleNamespace(nao=2, natom=2), result)


def test_timeline_keeps_valid_sample() -> None:
    work = dict.fromkeys(
        (
            "launches",
            "primitive_records",
            "xc_points",
            "grid_pair_visits",
            "tensor_executions",
            "tensor_work",
            "snapshot_export_work",
            "additional_device_peak_bound",
            "additional_host_numeric_bound",
            "endpoint_seconds",
        ),
        0,
    )
    work["timeline"] = {"exclusive_seconds": {"compute": 0.8}}
    result = SimpleNamespace(work=work, gradient=np.zeros((2, 3)))
    actual = _record("cold", 1.0, 0.2, SimpleNamespace(nao=2, natom=2), result)
    assert actual["status"] == "ok"
    assert actual["reconciliation"]["within_5_percent"]
