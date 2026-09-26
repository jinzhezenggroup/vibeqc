"""Campaign gates must reject invalid tolerances and retain zero-scale diagnostics."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import qualify_libxc_production_domain as campaign


@pytest.mark.parametrize("field", ("rtol", "atol"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), -1.0, True))
def test_campaign_rejects_invalid_tolerance(
    monkeypatch: pytest.MonkeyPatch, field: str, value: float
) -> None:
    args = SimpleNamespace(rtol=2.0e-6, atol=1.0e-8)
    setattr(args, field, value)
    monkeypatch.setattr(campaign, "_parse_args", lambda: args)
    with pytest.raises(ValueError, match="tolerances must be finite and nonnegative"):
        campaign.main()


def test_zero_absolute_tolerance_keeps_exact_zeros_serializable() -> None:
    error = campaign._relative_error(np.zeros(2), np.zeros(2), atol=0.0)
    assert error == 0.0
    json.dumps({"max_relative_error": error}, allow_nan=False)


def test_zero_reference_mismatch_has_no_finite_relative_error() -> None:
    error = campaign._relative_error(np.ones(2), np.zeros(2), atol=0.0)
    assert error is None
    json.dumps({"max_relative_error": error}, allow_nan=False)
