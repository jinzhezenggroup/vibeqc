from __future__ import annotations

import typing

import pytest

from benchmarks.check_scf_benchmark_stability import stability_summary


def _sample(iterations: int) -> dict[str, object]:
    return {"convergence": [{"iterations": iterations, "converged": True}]}


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


@pytest.mark.parametrize("bad", [True, 3.5, "3", -1, None])
def test_invalid_iteration_counts_cannot_collapse_to_a_shared_branch(
    bad: typing.Any,
) -> None:
    payload = _payload([3, 3], [3, 3])
    payload["vibeqc"]["warm_samples"][0]["convergence"][0]["iterations"] = bad
    with pytest.raises(ValueError, match="nonnegative integers"):
        stability_summary(payload)


def test_empty_system_records_do_not_form_a_valid_empty_branch() -> None:
    payload = _payload([3, 3], [3, 3])
    for engine in payload.values():
        for sample in engine["warm_samples"]:
            sample["convergence"] = []
    with pytest.raises(ValueError, match="nonempty convergence"):
        stability_summary(payload)


@pytest.mark.parametrize("converged", [False, None, 1])
def test_unconverged_or_unconfirmed_samples_are_diagnostic_only(
    converged: typing.Any,
) -> None:
    payload = _payload([3, 3], [3, 3])
    row = payload["vibeqc"]["warm_samples"][0]["convergence"][0]
    row["converged"] = converged
    summary = stability_summary(payload)
    assert not summary["headline_cross_engine_ratio_valid"]
    assert summary["ordinary_ratio_is_diagnostic_only"]
    assert any("confirmed convergence" in reason for reason in summary["reasons"])


def test_one_observation_cannot_establish_repeat_stability() -> None:
    summary = stability_summary(_payload([3], [3]))
    assert not summary["headline_cross_engine_ratio_valid"]
    assert (
        sum("at least two warm repeats" in reason for reason in summary["reasons"]) == 2
    )
