from __future__ import annotations

from benchmarks.check_scf_benchmark_stability import stability_summary


def _sample(iterations: int) -> dict[str, object]:
    return {"convergence": [{"iterations": iterations}]}


def _payload(vibeqc: list[int], gpu4pyscf: list[int]) -> dict[str, object]:
    return {
        "vibeqc": {"warm_samples": [_sample(value) for value in vibeqc]},
        "gpu4pyscf": {"warm_samples": [_sample(value) for value in gpu4pyscf]},
    }


def test_stable_shared_branch_supports_headline_ratio() -> None:
    summary = stability_summary(_payload([3, 3, 3, 3, 3], [3, 3, 3, 3, 3]))
    assert summary["headline_cross_engine_ratio_valid"] is True
    assert summary["shared_stable_branch"] == [3]
    assert summary["reasons"] == []


def test_unstable_reference_branch_is_inconclusive() -> None:
    summary = stability_summary(_payload([3, 3, 3, 3, 3], [1, 7, 4, 4, 1]))
    assert summary["headline_cross_engine_ratio_valid"] is False
    assert summary["gpu4pyscf"]["stable"] is False
    assert summary["gpu4pyscf"]["branch_histogram"] == {"1": 2, "4": 2, "7": 1}
    assert summary["ordinary_ratio_is_diagnostic_only"] is True


def test_stable_but_unmatched_branches_are_inconclusive() -> None:
    summary = stability_summary(_payload([3, 3, 3], [1, 1, 1]))
    assert summary["headline_cross_engine_ratio_valid"] is False
    assert summary["vibeqc"]["stable"] is True
    assert summary["gpu4pyscf"]["stable"] is True
    assert summary["shared_stable_branch"] is None
