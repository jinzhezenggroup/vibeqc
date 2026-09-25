"""Small real-endpoint CodSpeed suite for routine CPU performance regression."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

import pytest
from vibeqc import Calculator, GridSpec, KsOptions

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
    pr_fast: bool = False
    pr_extra: str | None = None
    grid_shape: tuple[int, int, int] | None = None


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
    # Keep the PR tier intentionally small. Two representative user endpoints
    # cost roughly two thirds of the current three-case simulation suite, while
    # covering direct HF and semilocal DFT on the same nontrivial molecule.
    _Case("h2-rhf-sto3g", "rhf", "sto-3g", _H2),
    _Case("water-rhf-sto3g", "rhf", "sto-3g", _WATER, pr_fast=True),
    _Case("water-pbe-sto3g", "pbe-rks", "sto-3g", _WATER, pr_fast=True),
    # WB97M-V is deliberately full-tier only. A small fixed grid keeps the
    # performance signal focused on the RSH/meta-GGA/VV10 execution path without
    # turning the CodSpeed job into another qualification-scale finite difference.
    _Case(
        "water-wb97mv-smallgrid-sto3g",
        "wb97m-v-rks",
        "sto-3g",
        _WATER,
        pr_extra="wb97mv",
        grid_shape=(12, 4, 8),
    ),
)


def _active_cases() -> tuple[_Case, ...]:
    tier = os.environ.get("VIBEQC_CODSPEED_TIER", "full")
    if tier == "full":
        return _CASES
    if tier == "pr":
        extras = frozenset(
            item.strip()
            for item in os.environ.get("VIBEQC_CODSPEED_EXTRA_CASES", "").split(",")
            if item.strip()
        )
        supported = frozenset(
            case.pr_extra for case in _CASES if case.pr_extra is not None
        )
        unknown = extras - supported
        if unknown:
            raise RuntimeError(
                "unknown VIBEQC_CODSPEED_EXTRA_CASES=" + ",".join(sorted(unknown))
            )
        return tuple(case for case in _CASES if case.pr_fast or case.pr_extra in extras)
    raise RuntimeError(f"unknown VIBEQC_CODSPEED_TIER={tier!r}")


def _calculator(case: _Case) -> Calculator:
    if case.grid_shape is None:
        return Calculator(method=case.method, basis=case.basis, device="cpu")

    radial, polar, azimuth = case.grid_shape
    return Calculator(
        method=case.method,
        basis=case.basis,
        device="cpu",
        ks_options=KsOptions(
            grid=GridSpec(
                radial_points=radial,
                angular_polar=polar,
                angular_azimuth=azimuth,
            )
        ),
        energy_tolerance=1e-10,
        density_tolerance=1e-8,
        max_iterations=160,
    )


@pytest.mark.parametrize("case", _active_cases(), ids=lambda case: case.name)
def test_cpu_warm_endpoint_walltime(
    benchmark: _BenchmarkFixture,
    case: _Case,
) -> None:
    """Track complete warm singlepoint endpoints on one-thread hosted CPU."""
    calculator = _calculator(case)

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
