"""Compile-time specialization and requested-output pruning for IntegralIR.

The pass consumes only facts already encoded in the mathematical request. It
does not inspect geometry, density, convergence state, or benchmark identity.
"""

from __future__ import annotations

from dataclasses import replace

from .blocks import RawBlock, SecondDerivative, WeightedDerivative
from .ir import ContractionSpec, IntegralIR, KernelConsumer


def _needs_derivative(
    contraction: object,
    *,
    source_has_derivative: bool,
) -> bool:
    """Return whether one retained output requires nuclear derivatives."""

    if isinstance(contraction, ContractionSpec):
        return contraction.kernel_consumer == KernelConsumer.FORCE
    if isinstance(contraction, (WeightedDerivative, SecondDerivative)):
        return True
    if isinstance(contraction, RawBlock):
        return source_has_derivative
    return source_has_derivative


def specialize_integral_ir(
    integral: IntegralIR,
    *,
    consumers: tuple[KernelConsumer | str, ...] | None = None,
    recurrence: str | None = None,
) -> IntegralIR:
    """Prune an IntegralIR to compile-time requested direct-HF outputs.

    consumers is an explicit output demand. Exact retained contraction records
    are preserved instead of being reconstructed from compatibility defaults.
    If pruning removes every derivative consumer, derivative intent is removed
    as well. Rys root count is then folded to the value-only order unless the
    caller explicitly selects another recurrence.

    Non-HF outputs are kept when consumers is omitted and are intentionally
    excluded when a direct FOCK/FORCE output subset is requested.
    """

    if not isinstance(integral, IntegralIR):
        raise TypeError("integral specialization requires IntegralIR")

    selected = integral.contractions
    if consumers is not None:
        normalized = tuple(KernelConsumer(item) for item in consumers)
        if not normalized:
            raise ValueError("integral specialization requires requested consumers")
        if len(set(normalized)) != len(normalized):
            raise ValueError("integral specialization contains duplicate consumers")
        available = integral.consumers
        missing = tuple(item for item in normalized if item not in available)
        if missing:
            names = ", ".join(item.value for item in missing)
            raise ValueError(
                f"integral specialization requests unavailable consumers: {names}"
            )
        demand = frozenset(normalized)
        selected = tuple(
            item for item in integral.contractions if item.kernel_consumer in demand
        )
        if not selected:
            raise ValueError("integral specialization removed every output")

    derivative = integral.derivative
    if derivative is not None and not any(
        _needs_derivative(item, source_has_derivative=True) for item in selected
    ):
        derivative = None

    selected_recurrence = integral.recurrence if recurrence is None else recurrence
    if (
        recurrence is None
        and integral.derivative is not None
        and derivative is None
        and selected_recurrence.startswith("rys")
    ):
        selected_recurrence = f"rys{integral.value_coulomb_order // 2 + 1}"

    if (
        selected == integral.contractions
        and derivative is integral.derivative
        and selected_recurrence == integral.recurrence
    ):
        return integral

    return replace(
        integral,
        derivative=derivative,
        contractions=selected,
        recurrence=selected_recurrence,
    )
