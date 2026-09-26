"""Production-owned method-neutral second-order orchestration for molecular HVPs.

This installed runtime module owns ordering, transactional assembly and
capability completeness.
It does not know HF, KS, functional names, CPHF/CPKS layouts, grids or integral
providers. A plan supplies a complete source inventory; adapters supply one
nuclear perturbation, one response solve and exactly one contributor per source.
"""

from __future__ import annotations

import typing
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from vibeqc_compiler.common.provenance import canonical_hash


def _identity(value: typing.Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} requires a nonempty identity")
    return value


def _cartesian(value: typing.Any, natoms: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.shape != (natoms, 3)
        or raw.dtype.kind not in "iuf"
        or np.iscomplexobj(raw)
        or not np.isfinite(raw).all()
    ):
        raise ValueError(f"{name} must be finite real with shape (natoms, 3)")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be representable in FP64")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class StationaryPerturbationProvider:
    """Build the method/state-specific nuclear perturbation for one direction."""

    identity: str
    build: typing.Callable[[np.ndarray], typing.Any]

    def __post_init__(self) -> None:
        _identity(self.identity, "perturbation provider")
        if not callable(self.build):
            raise TypeError("perturbation provider requires a callable build action")


@dataclass(frozen=True)
class StationaryResponseDriver:
    """Solve one opaque stationary response problem for a prepared perturbation."""

    identity: str
    solve: typing.Callable[[typing.Any], typing.Any]

    def __post_init__(self) -> None:
        _identity(self.identity, "response driver")
        if not callable(self.solve):
            raise TypeError("response driver requires a callable solve action")


@dataclass(frozen=True)
class StationaryHVPContext:
    """Shared immutable orchestration context seen by every source contributor."""

    plan_identity: str
    source_names: tuple[str, ...]
    direction: np.ndarray
    perturbation: typing.Any
    response: typing.Any


@dataclass(frozen=True)
class StationaryHVPContributor:
    """One complete source action in the plan-declared HVP decomposition."""

    source: str
    identity: str
    evaluate: typing.Callable[[StationaryHVPContext], typing.Any]

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("stationary-HVP contributor requires a source name")
        _identity(self.identity, "stationary-HVP contributor")
        if not callable(self.evaluate):
            raise TypeError("stationary-HVP contributor requires a callable action")


@dataclass(frozen=True, eq=False)
class StationarySecondOrderResult:
    """Detached complete HVP assembled only after every declared source succeeds."""

    direction: np.ndarray
    value: np.ndarray
    components: typing.Mapping[str, np.ndarray]
    response: typing.Any
    plan_identity: str
    identity: str
    diagnostics: typing.Mapping[str, typing.Any]


class StationarySecondOrderExecutor:
    """Execute one complete plan-defined HVP without method-specific dispatch.

    The plan is intentionally structural: it only needs a nonempty ``identity``
    and an ordered, unique ``source_names`` inventory. This lets the same
    orchestration consume MethodIR-derived DFT plans and existing/future HF
    second-order plans without teaching the executor either representation.
    """

    def __init__(
        self,
        plan: typing.Any,
        *,
        natoms: int,
        perturbation: StationaryPerturbationProvider,
        response: StationaryResponseDriver,
        contributors: typing.Iterable[StationaryHVPContributor],
    ) -> None:
        if type(natoms) is not int or natoms < 1:
            raise ValueError("stationary second-order execution requires natoms >= 1")
        self._natoms = natoms
        self._plan_identity = _identity(
            getattr(plan, "identity", None), "stationary second-order plan"
        )
        try:
            inventory = plan.source_names
            if isinstance(inventory, (str, bytes, AbstractSet)):
                raise TypeError("source_names must declare an ordered inventory")
            source_names = tuple(inventory)
        except (AttributeError, TypeError) as error:
            raise TypeError(
                "stationary second-order plan requires an ordered source_names inventory"
            ) from error
        if (
            not source_names
            or any(not isinstance(name, str) or not name for name in source_names)
            or len(set(source_names)) != len(source_names)
        ):
            raise ValueError(
                "stationary second-order plan source_names must be nonempty and unique"
            )
        if not isinstance(perturbation, StationaryPerturbationProvider):
            raise TypeError("expected StationaryPerturbationProvider")
        if not isinstance(response, StationaryResponseDriver):
            raise TypeError("expected StationaryResponseDriver")

        materialized = tuple(contributors)
        if not all(isinstance(item, StationaryHVPContributor) for item in materialized):
            raise TypeError("contributors require StationaryHVPContributor entries")
        names = tuple(item.source for item in materialized)
        if len(set(names)) != len(names):
            raise ValueError("stationary-HVP contributor source names must be unique")
        missing = tuple(name for name in source_names if name not in names)
        extra = tuple(name for name in names if name not in source_names)
        if missing or extra:
            raise ValueError(
                f"stationary-HVP contributor coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )

        by_name = {item.source: item for item in materialized}
        self._source_names = source_names
        self._perturbation = perturbation
        self._response = response
        self._contributors = tuple(by_name[name] for name in source_names)

    @property
    def plan_identity(self) -> str:
        return self._plan_identity

    @property
    def source_names(self) -> tuple[str, ...]:
        return self._source_names

    @property
    def identity(self) -> str:
        """Execution contract identity, excluding the perturbation direction."""
        return canonical_hash(
            {
                "schema": "vibeqc.stationary-second-order-executor/v1",
                "plan": self._plan_identity,
                "natoms": self._natoms,
                "sources": self._source_names,
                "perturbation_provider": self._perturbation.identity,
                "response_driver": self._response.identity,
                "contributors": [
                    {"source": item.source, "identity": item.identity}
                    for item in self._contributors
                ],
            }
        )

    def apply(self, direction: typing.Any) -> StationarySecondOrderResult:
        """Apply the complete stationary Hessian to one Cartesian direction.

        Nothing is published if perturbation construction, response, any source
        action or final finite-value validation fails. Contributors all receive
        the same response object and are executed in plan source order.
        """
        vector = _cartesian(direction, self._natoms, "HVP direction")
        perturbation = self._perturbation.build(vector)
        response = self._response.solve(perturbation)
        context = StationaryHVPContext(
            self._plan_identity,
            self._source_names,
            vector,
            perturbation,
            response,
        )

        components: dict[str, np.ndarray] = {}
        for contributor in self._contributors:
            components[contributor.source] = _cartesian(
                contributor.evaluate(context),
                self._natoms,
                f"{contributor.source} HVP contribution",
            )
        total = np.zeros((self._natoms, 3), dtype=np.float64)
        for source in self._source_names:
            total += components[source]
        if not np.isfinite(total).all():
            raise FloatingPointError("nonfinite stationary molecular HVP")
        total.setflags(write=False)

        result_identity = canonical_hash(
            {
                "executor": self.identity,
                "direction": vector.tolist(),
            }
        )
        published_components = MappingProxyType(dict(components))
        diagnostics = MappingProxyType(
            {
                "schema": "vibeqc.stationary-second-order-result/v1",
                "plan_identity": self._plan_identity,
                "executor_identity": self.identity,
                "source_order": self._source_names,
                "perturbation_provider": self._perturbation.identity,
                "response_driver": self._response.identity,
                "contributors": tuple(
                    (item.source, item.identity) for item in self._contributors
                ),
                "complete_source_coverage": True,
                "method_dispatch": "none",
            }
        )
        return StationarySecondOrderResult(
            vector,
            total,
            published_components,
            response,
            self._plan_identity,
            result_identity,
            diagnostics,
        )
