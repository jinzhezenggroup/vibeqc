"""The molecular HVP must reuse the native SCF-domain XC contributor."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

hvp = importlib.import_module("tools.vibeqc_hessian.rks_hvp")


def test_xc_binding_reuses_response_and_native_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = np.arange(6, dtype=float).reshape(2, 3)
    response = SimpleNamespace(direction=vector)
    operator = object()
    sources = SimpleNamespace(xc_ao=vector, xc_grid=2 * vector, xc_weight=3 * vector)
    calls = []

    def native(actual_operator: object, actual_response: object) -> object:
        calls.append((actual_operator, actual_response))
        return sources

    monkeypatch.setattr(hvp, "native_rks_xc_hvp_components", native)
    result = hvp._xc_hvp_components(operator, response, vector.copy())
    assert calls == [(operator, response)]
    assert tuple(result) == ("xc_ao", "xc_grid", "xc_weight")
    assert all(result[name] is getattr(sources, name) for name in result)


def test_xc_binding_rejects_a_different_solved_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object) -> object:
        pytest.fail("mismatched response reached the native contributor")

    monkeypatch.setattr(hvp, "native_rks_xc_hvp_components", forbidden)
    with pytest.raises(ValueError, match="direction"):
        hvp._xc_hvp_components(
            object(), SimpleNamespace(direction=np.zeros((1, 3))), np.ones((1, 3))
        )


def test_native_failure_is_not_replaced_by_generic_functional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object) -> object:
        raise ValueError("native point domain rejected")

    monkeypatch.setattr(hvp, "native_rks_xc_hvp_components", fail)
    vector = np.zeros((1, 3))
    with pytest.raises(ValueError, match="native point domain"):
        hvp._xc_hvp_components(object(), SimpleNamespace(direction=vector), vector)
