"""Pair-transfer budget admission precedes overlap, hashing and tensor conversion."""

from __future__ import annotations

import numpy as np
import pytest

from tools.vibeqc_local_cc import coupling
from tools.vibeqc_local_cc.spaces import PairSpace


def _space(rank: int = 3, virtual_rank: int = 3) -> PairSpace:
    return PairSpace(
        "reference",
        "localized",
        "domain",
        (0, 0),
        np.eye(virtual_rank)[:, virtual_rank - rank :],
        np.array([0.0] * (virtual_rank - rank) + [1.0] * rank),
        tuple(range(virtual_rank - rank, virtual_rank)),
        0.5,
        1e-12,
        False,
        False,
    )


@pytest.mark.parametrize("budget", [1, 359])
def test_budget_rejects_before_any_pair_transfer_or_tensor_conversion(
    monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    source = _space()
    calls = []

    def forbidden(*args: object, **kwargs: object) -> object:
        calls.append("transfer")
        raise AssertionError("transfer ran before admission")

    class Tensor:
        def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
            calls.append("input")
            raise AssertionError("input conversion ran before admission")

    monkeypatch.setattr(coupling, "pair_transfer", forbidden)
    with pytest.raises(MemoryError, match="numeric bytes"):
        coupling.project_pair_matrix(Tensor(), source, source, budget_bytes=budget)
    assert calls == []


def test_exact_declared_buffer_bound_accepts_independent_dense_projection() -> None:
    source = _space()
    values = np.arange(9.0).reshape(3, 3)
    # Source conversion dominates: (one 3x3 overlap + four 3x3 buffers) * 8.
    result = coupling.project_pair_matrix(values, source, source, budget_bytes=360)
    np.testing.assert_array_equal(result, values)
    assert not np.shares_memory(values, result)
    with pytest.raises(ValueError):
        result.setflags(write=True)


def test_small_pair_rank_still_charges_canonical_projector_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _space(1, 32)
    calls = []

    def forbidden(*args: object, **kwargs: object) -> object:
        calls.append(1)
        raise AssertionError("gauge work ran before admission")

    monkeypatch.setattr(coupling, "pair_transfer", forbidden)
    # One overlap element plus three full 32x32 portable projector buffers.
    with pytest.raises(MemoryError, match="24584 numeric bytes"):
        coupling.project_pair_matrix(
            np.ones((1, 1)), source, source, budget_bytes=24583
        )
    assert calls == []


@pytest.mark.parametrize("budget", [True, 0, -1, 2**63])
def test_invalid_budget_never_constructs_transfer(
    monkeypatch: pytest.MonkeyPatch, budget: object
) -> None:
    source = _space()
    calls = []

    def forbidden(*args: object, **kwargs: object) -> object:
        calls.append(1)
        raise AssertionError("transfer ran before invalid-budget check")

    monkeypatch.setattr(coupling, "pair_transfer", forbidden)
    with pytest.raises(ValueError, match="numeric budget"):
        coupling.project_pair_matrix(np.eye(3), source, source, budget_bytes=budget)
    assert calls == []
