"""Small real-endpoint CodSpeed suite for routine CPU performance regression."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

import pytest
from vibeqc import Calculator

if TYPE_CHECKING:
    from collections.abc import Callable

_T = TypeVar("_T")
Atom = tuple[str, tuple[float, float, float]]


class _BenchmarkFixture(Protocol):
    def __call__(self, target: Callable[[], _T]) -> _T: ...


@dataclass(frozen=True)
class _Case:
    name: str
    method: str
    basis: str
    atoms: tuple[Atom, ...]


_H2 = (
    ("H", (0.0, 0.0, -0.7)),
    ("H", (0.0, 0.0, 0.7)),
)
_WATER = (
    ("O", (0.0, 0.0, 0.0)),
    ("H", (1.43233673, 0.0, 1.10715266)),
    ("H", (-1.43233673, 0.0, 1.10715266)),
)
_CASES = (
    _Case("h2-rhf-sto3g", "rhf", "sto-3g", _H2),
    _Case("water-rhf-sto3g", "rhf", "sto-3g", _WATER),
    _Case("water-pbe-sto3g", "pbe-rks", "sto-3g", _WATER),
)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_cpu_warm_endpoint_walltime(
    benchmark: _BenchmarkFixture,
    case: _Case,
) -> None:
    """Track complete warm singlepoint endpoints on one-thread hosted CPU."""
    calculator = Calculator(method=case.method, basis=case.basis, device="cpu")

    warmup = calculator.singlepoint(case.atoms, properties=("energy",))
    assert warmup.converged
    assert math.isfinite(warmup.energy)

    def run_once() -> float:
        result = calculator.singlepoint(case.atoms, properties=("energy",))
        if not result.converged:
            raise RuntimeError(f"{case.name} did not converge")
        if not math.isfinite(result.energy):
            raise RuntimeError(f"{case.name} returned a non-finite energy")
        return result.energy

    energy = benchmark(run_once)
    assert math.isfinite(energy)
