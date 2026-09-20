"""Finite compiler schedules and independently checked permutation semantics."""

import typing
from itertools import product

import numpy as np
import pytest
from vibeqc_compiler.integral import first_derivative_schedule as schedule

DOMAIN = ("", "x", "xx", "xy", "xz", "y", "yy", "yz", "z", "zz")


def test_complete_spd_schedule_is_finite_and_deterministic() -> None:
    requests = schedule.derivative_requests(DOMAIN)
    assert len(requests) == 362
    assert sum(op == "four_center_eri" for op, _ in requests) == 313
    assert requests == tuple(sorted(set(requests)))
    assert schedule.derivative_requests(DOMAIN) is requests


@pytest.mark.parametrize(
    "operator", ["overlap", "kinetic", "nuclear_attraction", "four_center_eri"]
)
def test_monomials_exponents_and_center_pairs_survive_mapping(operator: str) -> None:
    rank = 4 if operator == "four_center_eri" else 2
    # Unequal coordinates/exponents expose inverted permutations. The scalar
    # monomial-Gaussian factors are independent of the derivative IR/emitter.
    centers = np.array(
        [
            [0.17, -0.38, 0.61],
            [0.83, 0.23, -0.74],
            [-0.47, 0.91, 0.32],
            [0.52, -0.16, 1.13],
        ]
    )
    points = np.array([[0.76, 0.34, 0.97], [-0.28, 0.51, 0.81]])
    exponents = np.array([0.37, 0.71, 1.09, 1.43])

    def scalar(
        labels: tuple[str, ...], xyz: np.ndarray, p: np.ndarray, alpha: np.ndarray
    ) -> float:
        value = 1.0
        for i, label in enumerate(labels):
            delta = p[i // 2] - xyz[i]
            value *= np.exp(-alpha[i] * np.dot(delta, delta))
            for axis in label:
                value *= delta["xyz".index(axis)]
        return value

    for labels in product(DOMAIN, repeat=rank):
        binding = schedule.derivative_binding(operator, labels)
        order = binding.centers[:rank]
        assert sorted(order) == list(range(rank))
        # ERI centers may swap within each electron pair, or swap both pairs.
        electron_order = [order[i] // 2 for i in range(0, rank, 2)]
        mapped = scalar(
            binding.request[1],
            centers[np.ix_(order, binding.axes)],
            points[np.ix_(electron_order, binding.axes)],
            exponents[list(order)],
        )
        expected = scalar(labels, centers[:rank], points, exponents[:rank])
        assert abs(mapped - expected) < 1e-15
        if operator == "nuclear_attraction":
            assert binding.centers[2] == 2


@pytest.mark.parametrize(
    "domain", [(), ("x", ""), ("x", "x"), ("xxx",), ("yx",), ("q",)]
)
def test_invalid_component_domain_is_rejected(domain: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        schedule.derivative_requests(domain)


@pytest.mark.parametrize(
    "operator,components", [("bad", ("", "")), ("overlap", ("",)), ("nuclear", ("x",))]
)
def test_invalid_operator_and_arity(operator: str, components: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        schedule.derivative_binding(operator, components)
    binding = schedule.derivative_binding("nuclear", ())
    assert binding.centers == (0, 1) and binding.axes == (0, 1, 2)


def test_source_units_and_byte_caps(monkeypatch: typing.Any) -> None:
    schedule.derivative_sources.cache_clear()
    monkeypatch.setattr(
        schedule, "emit_first_derivative_cpu", lambda requests: "x" * len(requests)
    )
    try:
        sources = schedule.derivative_sources(DOMAIN)
        assert len(sources) == 46
        assert all(len(requests) <= 8 for requests, _ in sources)
        assert sum(len(source) for _, source in sources) == 362
        assert schedule.derivative_sources(DOMAIN) is sources
        for limit in ("MAX_UNIT_BYTES", "MAX_PROGRAM_BYTES"):
            schedule.derivative_sources.cache_clear()
            with monkeypatch.context() as patch:
                patch.setattr(schedule, limit, 7)
                with pytest.raises(ValueError, match="source budget"):
                    schedule.derivative_sources(DOMAIN)
    finally:
        schedule.derivative_sources.cache_clear()
