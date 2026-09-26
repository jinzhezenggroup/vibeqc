"""Compile canonical MethodIR into a method-name-free KS contribution plan.

This module owns scientific composition only. Backend/provider admission remains
separate: a plan may be complete while a CPU/CUDA lowerer for one primitive is
still unavailable. That separation lets new methods inherit execution as their
primitive families become executable without adding named-method branches.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from fractions import Fraction

from vibeqc_compiler.common.provenance import canonical_hash

from .dispersion import DispersionCorrectionPrimitive
from .gcp import GeometricCounterpoisePrimitive
from .nonlocal_correlation import NonlocalCorrelationPrimitive
from .spec import (
    FULL_RANGE,
    LONG_RANGE,
    SHORT_RANGE,
    ExactExchangePrimitive,
    MethodIR,
    RangeSeparatedExchangePrimitive,
    SemilocalXCPrimitive,
    UnsupportedMethod,
)

KS_EXECUTION_PLAN_VERSION = "ks-execution-plan-v1"


@dataclass(frozen=True)
class KsExchangeContribution:
    """One exact-exchange operator contribution in the target density convention."""

    operator: str
    coefficient: Fraction
    omega: Fraction
    fock_coefficient: Fraction

    def semantic_payload(self) -> typing.Any:
        return {
            "operator": self.operator,
            "coefficient": str(self.coefficient),
            "omega": str(self.omega),
            "fock_coefficient": str(self.fock_coefficient),
        }


@dataclass(frozen=True)
class KsExecutionPlan:
    """Backend-neutral self-consistent KS projection of one MethodIR graph."""

    method: MethodIR
    semilocal: SemilocalXCPrimitive
    exchange: tuple[KsExchangeContribution, ...]
    nonlocal_correlation: NonlocalCorrelationPrimitive | None
    post_scf: tuple[DispersionCorrectionPrimitive | GeometricCounterpoisePrimitive, ...]
    version: str = KS_EXECUTION_PLAN_VERSION

    @property
    def required_lowerers(self) -> tuple[str, ...]:
        lowerers = ["semilocal-xc"]
        lowerers.extend(f"{term.operator}-exchange" for term in self.exchange)
        if self.nonlocal_correlation is not None:
            lowerers.append("nonlocal-correlation")
        return tuple(lowerers)

    @property
    def self_consistent(self) -> bool:
        semilocal_ok = "feature-gradient" in self.semilocal.derivative_capabilities
        exchange_ok = all(
            term.operator in (FULL_RANGE, SHORT_RANGE, LONG_RANGE)
            for term in self.exchange
        )
        nonlocal_ok = (
            self.nonlocal_correlation is None
            or "ks-potential" in self.nonlocal_correlation.derivative_capabilities
        )
        return semilocal_ok and exchange_ok and nonlocal_ok

    def semantic_payload(self) -> typing.Any:
        return {
            "version": self.version,
            "method_identity": self.method.identity,
            "spin": self.method.spin,
            "reference": self.method.reference,
            "semilocal": self.semilocal.semantic_payload(),
            "exchange": [term.semantic_payload() for term in self.exchange],
            "nonlocal_correlation": (
                self.nonlocal_correlation.semantic_payload()
                if self.nonlocal_correlation is not None
                else None
            ),
            "post_scf": [primitive.semantic_payload() for primitive in self.post_scf],
            "required_lowerers": self.required_lowerers,
        }

    def to_payload(self) -> typing.Any:
        return {
            **self.semantic_payload(),
            "method_identifier": self.method.identifier,
            "method_manifest_identity": self.method.manifest_identity,
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.semantic_payload())


def _exchange_contribution(
    primitive: ExactExchangePrimitive | RangeSeparatedExchangePrimitive, spin: str
) -> KsExchangeContribution:
    divisor = 2 if spin == "unpolarized" else 1
    omega = (
        Fraction(0)
        if isinstance(primitive, ExactExchangePrimitive)
        else primitive.omega
    )
    return KsExchangeContribution(
        primitive.operator,
        primitive.coefficient,
        omega,
        -primitive.coefficient / divisor,
    )


def compile_ks_execution_plan(method: MethodIR) -> KsExecutionPlan:
    """Project MethodIR primitives into one inspectable self-consistent KS plan."""
    if not isinstance(method, MethodIR):
        raise TypeError("KS execution planning requires a resolved MethodIR")

    semilocal = tuple(
        p for p in method.primitives if isinstance(p, SemilocalXCPrimitive)
    )
    if len(semilocal) != 1:
        raise UnsupportedMethod(
            "self-consistent KS execution requires exactly one semilocal XC primitive"
        )

    exchange_primitives = tuple(
        p
        for p in method.primitives
        if isinstance(p, (ExactExchangePrimitive, RangeSeparatedExchangePrimitive))
    )
    exchange = tuple(
        _exchange_contribution(primitive, method.spin)
        for primitive in exchange_primitives
    )

    nonlocal_primitives = tuple(
        p for p in method.primitives if isinstance(p, NonlocalCorrelationPrimitive)
    )
    if len(nonlocal_primitives) > 1:
        raise UnsupportedMethod(
            "KS execution accepts at most one nonlocal correlation primitive"
        )
    nonlocal_correlation = nonlocal_primitives[0] if nonlocal_primitives else None

    post_scf = tuple(
        p
        for p in method.primitives
        if isinstance(
            p, (DispersionCorrectionPrimitive, GeometricCounterpoisePrimitive)
        )
    )
    consumed = (
        len(semilocal)
        + len(exchange_primitives)
        + len(nonlocal_primitives)
        + len(post_scf)
    )
    if consumed != len(method.primitives):
        raise UnsupportedMethod(
            "MethodIR contains a primitive without a KS execution role"
        )

    range_terms = tuple(
        term for term in exchange if term.operator in (SHORT_RANGE, LONG_RANGE)
    )
    range_omegas = {term.omega for term in range_terms}
    if len(range_omegas) > 1:
        raise UnsupportedMethod(
            "range-separated KS exchange terms require one shared omega"
        )
    functional_omega = semilocal[0].functional.range_omega
    if range_omegas and functional_omega and functional_omega not in range_omegas:
        raise UnsupportedMethod(
            "semilocal attenuation and exact range exchange disagree on omega"
        )

    plan = KsExecutionPlan(
        method=method,
        semilocal=semilocal[0],
        exchange=exchange,
        nonlocal_correlation=nonlocal_correlation,
        post_scf=post_scf,
    )
    if not plan.self_consistent:
        raise UnsupportedMethod(
            "MethodIR lacks a complete self-consistent KS potential"
        )
    return plan
