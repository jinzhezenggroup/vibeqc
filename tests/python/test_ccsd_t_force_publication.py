"""Force-batch publication shares the project's immutable finite-real contract."""

from types import SimpleNamespace

import numpy as np
import pytest

from tools.vibeqc_cc import ccsd_t_api as api


def test_force_batch_owns_irreversibly_read_only_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = np.arange(6.0).reshape(2, 3)
    monkeypatch.setattr(
        api, "rccsd_t_force", lambda *args, **kwargs: SimpleNamespace(forces=raw)
    )
    sources = [SimpleNamespace(nbf=2, electron_count=2) for _ in range(2)]
    result = api.rccsd_t_batch_forces(sources)
    for item in result.items:
        assert item.converged
        assert item.forces is not None
        assert not np.shares_memory(item.forces, raw)
        with pytest.raises(ValueError):
            item.forces.setflags(write=True)
    raw[:] = 100
    for item in result.items:
        np.testing.assert_array_equal(item.forces, np.arange(6.0).reshape(2, 3))


@pytest.mark.parametrize(
    "invalid",
    (
        np.array([[1.0 + 2.0j, 0.0, 0.0]]),
        np.empty((0, 3)),
        np.array([[np.nan, 0.0, 0.0]]),
        np.array([[np.inf, 0.0, 0.0]]),
        np.ones((1, 2)),
        np.ones(3),
    ),
    ids=("complex", "empty", "nan", "inf", "wrong-width", "wrong-rank"),
)
def test_malformed_force_item_does_not_publish_or_block_its_neighbor(
    monkeypatch: pytest.MonkeyPatch,
    invalid: np.ndarray,
) -> None:
    def endpoint(source: SimpleNamespace, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(forces=invalid if source.bad else np.ones((1, 3)))

    monkeypatch.setattr(api, "rccsd_t_force", endpoint)
    sources = [
        SimpleNamespace(nbf=2, electron_count=2, bad=bad) for bad in (True, False)
    ]
    result = api.rccsd_t_batch_forces(sources)
    assert result.items[0].status == "error"
    assert not result.items[0].converged
    assert result.items[0].forces is None and result.items[0].result is None
    assert result.items[1].converged and result.items[1].index == 1
    np.testing.assert_array_equal(result.items[1].forces, np.ones((1, 3)))
