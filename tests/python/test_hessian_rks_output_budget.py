"""Exercise Hessian publication accounting without a native SCF dependency."""

from __future__ import annotations

import typing
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from tools.vibeqc_hessian import rks_molecular


def _stub_operator(monkeypatch: pytest.MonkeyPatch) -> typing.Any:
    class Operator:
        xc_kernel = SimpleNamespace(basis=SimpleNamespace(natom=2))
        state = SimpleNamespace(
            identity=SimpleNamespace(
                method="lda-rks",
                to_payload=lambda: {"fixture": "output-budget-only"},
            )
        )

        def validate_current(self) -> None:
            pass

    monkeypatch.setattr(rks_molecular, "NativeRKSResponse", Operator)
    monkeypatch.setattr(rks_molecular, "_checked_plan", lambda _: None)
    return Operator()


def test_symmetry_check_reuses_scratch_before_immutable_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _stub_operator(monkeypatch)
    expected = np.arange(36, dtype=np.float64).reshape(6, 6)
    expected_error = float(np.max(np.abs(expected - expected.T)))
    calls = []
    scratch_refs = []
    original_abs = np.abs
    original_immutable = rks_molecular.immutable

    def fake_many(
        _operator: typing.Any,
        directions: typing.Any,
        **kwargs: typing.Any,
    ) -> SimpleNamespace:
        vectors = np.asarray(directions).reshape(len(directions), 6)
        calls.append(len(directions))
        return SimpleNamespace(
            values=(vectors @ expected.T).reshape(len(directions), 2, 3),
            identity=f"test-block-{len(calls)}",
            diagnostics={"multi_rhs_calls": 1},
        )

    def checked_abs(
        value: typing.Any,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        if np.shape(value) == expected.shape:
            assert kwargs.get("out") is value, "second dense scratch allocation"
            scratch_refs.append(weakref.ref(value))
        return original_abs(value, *args, **kwargs)

    def checked_immutable(value: typing.Any, **kwargs: typing.Any) -> typing.Any:
        assert scratch_refs, "symmetry check was not exercised"
        assert all(ref() is None for ref in scratch_refs)
        return original_immutable(value, **kwargs)

    monkeypatch.setattr(rks_molecular, "rks_hvp_many", fake_many)
    monkeypatch.setattr(rks_molecular.np, "abs", checked_abs)
    monkeypatch.setattr(rks_molecular, "immutable", checked_immutable)
    result = rks_molecular.rks_hessian(
        operator,
        block_size=2,
        output_budget_bytes=2 * expected.nbytes,
    )
    monkeypatch.setattr(rks_molecular.np, "abs", original_abs)

    np.testing.assert_array_equal(result.matrix, expected)
    assert calls == [2, 2, 2]
    assert result.diagnostics["raw_symmetry_error"] == expected_error
    assert result.diagnostics["output_peak_bound_bytes"] == 2 * expected.nbytes
    assert not result.matrix.flags.writeable


def test_one_byte_short_output_budget_refuses_before_block_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _stub_operator(monkeypatch)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("HVP started before output admission")

    monkeypatch.setattr(rks_molecular, "rks_hvp_many", forbidden)
    with pytest.raises(ValueError, match="output_budget_bytes"):
        rks_molecular.rks_hessian(
            operator,
            block_size=2,
            output_budget_bytes=2 * 6 * 6 * np.dtype(np.float64).itemsize - 1,
        )
