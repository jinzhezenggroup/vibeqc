"""CodSpeed walltime coverage for real VibeQC CUDA endpoints on CuMetal/Metal."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

import pytest
from vibeqc import Calculator

from benchmarks._cases import BenchmarkCase, benchmark_cases

if TYPE_CHECKING:
    from collections.abc import Callable

_T = TypeVar("_T")


class _BenchmarkFixture(Protocol):
    def __call__(self, target: Callable[[], _T]) -> _T: ...


@dataclass(frozen=True)
class _Endpoint:
    name: str
    case_name: str
    properties: tuple[str, ...]


_ENDPOINTS = (
    _Endpoint(
        "water-24ao-rhf-energy",
        "water-def2-svp-spherical",
        ("energy",),
    ),
    _Endpoint(
        "water-24ao-rhf-energy-forces",
        "water-def2-svp-spherical",
        ("energy", "forces"),
    ),
    _Endpoint(
        "water-tetramer-96ao-rhf-energy-forces",
        "water-tetramer-def2-svp-spherical",
        ("energy", "forces"),
    ),
)


def _calculator(case: BenchmarkCase) -> Calculator:
    return Calculator(
        method=case.method,
        basis=case.vibeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
    )


def _run(
    calculator: Calculator,
    case: BenchmarkCase,
    properties: tuple[str, ...],
) -> float:
    result = calculator.singlepoint(
        case.atoms,
        charge=case.charge,
        multiplicity=case.multiplicity,
        properties=properties,
    )
    if result.executed_backend != "cuda":
        raise RuntimeError(
            f"CuMetal endpoint unexpectedly executed on {result.executed_backend!r}"
        )
    if not result.converged:
        raise RuntimeError("CuMetal endpoint did not converge")
    if not math.isfinite(result.energy):
        raise RuntimeError("CuMetal endpoint returned a non-finite energy")
    if "forces" in properties:
        if result.forces is None:
            raise RuntimeError("CuMetal force endpoint returned no forces")
        if not all(math.isfinite(component) for force in result.forces for component in force):
            raise RuntimeError("CuMetal endpoint returned non-finite forces")
    elif result.forces is not None:
        raise RuntimeError("energy-only CuMetal endpoint unexpectedly computed forces")
    return result.energy


@pytest.mark.parametrize("endpoint", _ENDPOINTS, ids=lambda endpoint: endpoint.name)
def test_cumetal_real_endpoint_walltime(
    benchmark: _BenchmarkFixture,
    endpoint: _Endpoint,
) -> None:
    """Track complete warm VibeQC CUDA endpoints executing through CuMetal/Metal."""

    case = benchmark_cases()[endpoint.case_name]
    calculator = _calculator(case)

    # Keep compilation, first-use runtime setup, and cold allocator effects outside
    # the measured region. The repeated sample remains the public Calculator path.
    _run(calculator, case, endpoint.properties)

    energy = benchmark(lambda: _run(calculator, case, endpoint.properties))
    assert math.isfinite(energy)
