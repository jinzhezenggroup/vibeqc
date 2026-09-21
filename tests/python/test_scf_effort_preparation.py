"""Partial preparation is transactional across all SCF effort owners."""

from types import SimpleNamespace

import pytest

from tools.vibeqc_numerics import scf_effort_geomopt_benchmark as bench


@pytest.mark.parametrize("failed_index", (1, 2))
def test_failed_level_preparation_closes_prior_owners(
    monkeypatch: pytest.MonkeyPatch, failed_index: int
) -> None:
    entered, closed = [], []

    class Owner:
        def __init__(self, index: int) -> None:
            self.index = index

        def __enter__(self) -> object:
            entered.append(self.index)
            return self

        def __exit__(self, *args: object) -> None:
            closed.append(self.index)

    count = [0]

    def calculator(*args: object, **kwargs: object) -> object:
        index = count[0]
        count[0] += 1
        if index == failed_index:
            raise RuntimeError("prepare failure")
        return SimpleNamespace(prepare_batch=lambda *a, **k: Owner(index))

    monkeypatch.setattr(bench, "_calculator", calculator)
    case = SimpleNamespace(atoms=(), charge=0, multiplicity=1)
    with (
        pytest.raises(RuntimeError, match="prepare failure"),
        bench.PreparedLevels(case, "cpu", "fp64"),
    ):
        raise AssertionError("unreachable")
    assert closed == list(reversed(entered))


@pytest.mark.parametrize("cleanup_failed_index", (0, 1))
def test_cleanup_failure_preserves_preparation_error_and_clears_owner_map(
    monkeypatch: pytest.MonkeyPatch, cleanup_failed_index: int
) -> None:
    entered, closed = [], []

    class Owner:
        def __init__(self, index: int) -> None:
            self.index = index

        def __enter__(self) -> object:
            entered.append(self.index)
            return self

        def __exit__(self, *args: object) -> None:
            closed.append(self.index)
            if self.index == cleanup_failed_index:
                raise RuntimeError(f"cleanup failure {self.index}")

    count = [0]

    def calculator(*args: object, **kwargs: object) -> object:
        index = count[0]
        count[0] += 1
        if index == 2:
            raise RuntimeError("prepare failure")
        return SimpleNamespace(prepare_batch=lambda *a, **k: Owner(index))

    monkeypatch.setattr(bench, "_calculator", calculator)
    case = SimpleNamespace(atoms=(), charge=0, multiplicity=1)
    prepared = bench.PreparedLevels(case, "cpu", "fp64")
    with pytest.raises(RuntimeError, match="prepare failure") as caught:
        prepared.__enter__()
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "cleanup failure" in str(caught.value.__cause__)
    assert closed == [1, 0]
    assert prepared._batches == {}
