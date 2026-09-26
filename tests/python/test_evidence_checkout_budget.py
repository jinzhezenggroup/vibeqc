"""Many small reports cannot bypass the benchmark checkout budget."""

import typing

import pytest

from tools.vibeqc_validation.retention import check


def policy(budget: typing.Any) -> typing.Any:
    return {
        "schema": "vibeqc.retention-policy.v1",
        "exceptions": {},
        "review_size_bytes": 1024,
        "benchmark_results_max_bytes": budget,
    }


def test_budget_sums_small_reports_but_not_permanent_test_fixtures() -> None:
    blobs = {
        "benchmarks/results/a.json": b"1234",
        "benchmarks/results/b.json": b"5678",
        "tests/reference_data/oracle.json": b"0" * 100,
    }
    assert check(blobs, policy(8)) == []
    errors = check(blobs, policy(7))
    assert len(errors) == 1
    assert "8 bytes exceeds 7-byte aggregate budget" in errors[0]


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "1024"])
def test_invalid_budget_is_not_a_size_guard_escape(budget: typing.Any) -> None:
    with pytest.raises(ValueError, match="benchmark_results_max_bytes"):
        check({}, policy(budget))
