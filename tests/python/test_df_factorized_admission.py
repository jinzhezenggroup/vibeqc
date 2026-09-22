"""Budget admission precedes factor-sized numerical temporaries."""

import numpy as np
import pytest

from tools.vibeqc_cc import df_factorized as owner


def inputs() -> tuple[np.ndarray, ...]:
    return (
        np.zeros((7, 2, 3)),
        np.zeros((7, 3, 3)),
        np.zeros((2, 3)),
        np.zeros((2, 2, 3, 3)),
    )


def test_insufficient_budget_rejects_before_numerical_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = inputs()
    monkeypatch.setattr(
        owner.np,
        "isfinite",
        lambda value: pytest.fail("factor-sized validation before admission"),
    )
    with pytest.raises(MemoryError, match="numeric bytes"):
        owner.virtual_corrections(*values, max_bytes=1)


def test_factor_checks_do_not_allocate_over_the_auxiliary_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = inputs()
    original = np.isfinite

    def finite(value: np.ndarray) -> np.ndarray:
        assert value.shape not in (values[0].shape, values[1].shape)
        return original(value)

    monkeypatch.setattr(owner.np, "isfinite", finite)
    result = owner.virtual_corrections(
        *values, max_bytes=owner.virtual_correction_workspace_bytes(2, 3)
    )
    assert all(np.count_nonzero(x) == 0 for x in result)
