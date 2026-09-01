"""Generic fused shell-class recurrence schedules and host-side oracle.

This module mirrors the arithmetic shape required by a high-performance CUDA
kernel: one lane owns one Cartesian AO quartet, primitive geometry and Coulomb
states are shared, and all xyz derivatives are accumulated together.  It is
kept independent from CUDA text emission so every generated shell class can be
checked against the symbolic AD lowering before any production integration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .cuda_schedule import (
    KernelIR,
    ScheduleIR,
    default_schedule,
)
from .cuda_target import DEFAULT_CUDA_TARGET, CudaTargetInfo
from .ir import IntegralIR, KernelConsumer, build_integral_ir
from .shell_spec import AXES, ShellClassSpec

ShellComponent = tuple[str, str, str, str]
CoulombState = tuple[int, int, int]
_AXIS_INDEX = {axis: index for index, axis in enumerate(AXES)}


@dataclass(frozen=True, slots=True)
class FusedShellPlan:
    """Static component, Coulomb, and cooperative-block schedule."""

    kernel: KernelIR
    spec: ShellClassSpec
    components: tuple[ShellComponent, ...]
    coulomb_states: tuple[CoulombState, ...]
    coulomb_indices: tuple[int, ...]
    block_threads: int

    @property
    def warp_count(self) -> int:
        """Return the number of full warps used by one generated block."""

        return self.kernel.schedule.warp_count

    @property
    def schedule(self) -> ScheduleIR:
        """Return the explicit execution policy consumed by CUDA emission."""

        return self.kernel.schedule


@dataclass(frozen=True, slots=True)
class FusedShellResult:
    """Value and all-center gradient produced by the shared recurrence."""

    value: float
    gradients: tuple[tuple[float, float, float], ...]


def build_fused_shell_plan(
    spec: ShellClassSpec,
    *,
    consumers: tuple[KernelConsumer | str, ...] | None = None,
    schedule: ScheduleIR | None = None,
    recurrence: str | None = None,
    target: CudaTargetInfo = DEFAULT_CUDA_TARGET,
    integral: IntegralIR | None = None,
) -> FusedShellPlan:
    """Lower integral and schedule IRs into deterministic CUDA lookup tables.

    ``integral`` lets compiler stages carry an explicit derivative and
    contraction specification all the way into scheduling.  Legacy callers
    may continue to provide ``consumers`` and ``recurrence``; when both forms
    are supplied, they must agree rather than silently rebuilding the IR.
    """

    if integral is None:
        selected_consumers = (KernelConsumer.FORCE,) if consumers is None else consumers
        selected_recurrence = "subset_wick" if recurrence is None else recurrence
        integral = build_integral_ir(
            spec,
            selected_consumers,
            recurrence=selected_recurrence,
        )
    else:
        if integral.spec != spec:
            raise ValueError("fused shell plan spec does not match its integral IR")
        if consumers is not None:
            normalized = frozenset(KernelConsumer(item) for item in consumers)
            if normalized != set(integral.consumers):
                raise ValueError("consumer and integral specifications disagree")
        if recurrence is not None and recurrence != integral.recurrence:
            raise ValueError("recurrence and integral specifications disagree")
    selected_schedule = schedule or default_schedule(integral, target)
    kernel = KernelIR(
        integral=integral,
        schedule=selected_schedule,
        target=target,
    )
    maximum_order = integral.maximum_coulomb_order
    states = tuple(
        (x_order, y_order, total - x_order - y_order)
        for total in range(maximum_order + 1)
        for x_order in range(total + 1)
        for y_order in range(total - x_order + 1)
    )
    state_indices = {state: index for index, state in enumerate(states)}
    side = maximum_order + 1
    dense_indices = tuple(
        state_indices.get((x_order, y_order, z_order), -1)
        for x_order in range(side)
        for y_order in range(side)
        for z_order in range(side)
    )
    return FusedShellPlan(
        kernel=kernel,
        spec=spec,
        components=spec.components,
        coulomb_states=states,
        coulomb_indices=dense_indices,
        block_threads=selected_schedule.block_threads,
    )


def _axis_wick_multiplicity(order: int, pairs: int) -> int:
    """Return the number of closed same-axis Coulomb contractions."""

    numerator = 1
    for value in range(order - 2 * pairs + 1, order + 1):
        numerator *= value
    denominator = 2**pairs
    for value in range(2, pairs + 1):
        denominator *= value
    return numerator // denominator


def _coulomb_value(state: CoulombState, variables: Mapping[str, float]) -> float:
    """Evaluate one Cartesian Coulomb derivative from shared geometry."""

    x_order, y_order, z_order = state
    total_order = sum(state)
    rho = variables["rho"]
    value = 0.0
    for x_pairs in range(x_order // 2 + 1):
        for y_pairs in range(y_order // 2 + 1):
            for z_pairs in range(z_order // 2 + 1):
                contraction_count = x_pairs + y_pairs + z_pairs
                boys_order = total_order - contraction_count
                multiplicity = (
                    _axis_wick_multiplicity(x_order, x_pairs)
                    * _axis_wick_multiplicity(y_order, y_pairs)
                    * _axis_wick_multiplicity(z_order, z_pairs)
                )
                value += (
                    multiplicity
                    * (-2.0 * rho) ** boys_order
                    * variables["difference_x"] ** (x_order - 2 * x_pairs)
                    * variables["difference_y"] ** (y_order - 2 * y_pairs)
                    * variables["difference_z"] ** (z_order - 2 * z_pairs)
                    * variables[f"boys_{boys_order}"]
                )
    return value


def _matching_masks(axes: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Enumerate all disjoint same-axis Wick matchings.

    Distinct pairings can have the same removed mask (for example the three
    perfect matchings of four x quanta).  Those duplicates are intentional:
    summing them produces the exact Wick multiplicity without handwritten
    order-specific cases.
    """

    def visit(indices: tuple[int, ...]):
        if not indices:
            yield (0, 0)
            return
        first = indices[0]
        rest = indices[1:]
        yield from visit(rest)
        for position, second in enumerate(rest):
            if axes[first] != axes[second]:
                continue
            remaining = rest[:position] + rest[position + 1 :]
            for removed, count in visit(remaining):
                yield (
                    removed | (1 << first) | (1 << second),
                    count + 1,
                )

    # Sorting preserves the legacy order-2/order-3 accumulation order while
    # extending deterministically to double and higher disjoint matchings.
    return tuple(
        sorted(
            visit(tuple(range(len(axes)))),
            key=lambda item: (item[1], item[0]),
        )
    )


def _pair_terms(
    axes: Sequence[int],
    shifts: Sequence[float],
    shift_gradients: Sequence[float],
    inverse_two_exponent: float,
) -> tuple[tuple[CoulombState, float, tuple[float, float, float]], ...]:
    """Build the subset/Wick coefficient schedule for one Gaussian pair."""

    terms = []
    for subset in range(1 << len(axes)):
        state = tuple(
            sum(
                bool(subset & (1 << quantum)) and axes[quantum] == axis
                for quantum in range(len(axes))
            )
            for axis in range(3)
        )
        coefficient = 0.0
        gradient = [0.0, 0.0, 0.0]
        for removed, contraction_count in _matching_masks(axes):
            if subset & removed:
                continue
            inverse_factor = inverse_two_exponent ** (
                subset.bit_count() + contraction_count
            )
            surviving = [
                quantum
                for quantum in range(len(axes))
                if ((subset | removed) & (1 << quantum)) == 0
            ]
            matching_coefficient = inverse_factor
            for quantum in surviving:
                matching_coefficient *= shifts[quantum]
            coefficient += matching_coefficient
            for differentiated in surviving:
                derivative = inverse_factor * shift_gradients[differentiated]
                for quantum in surviving:
                    if quantum != differentiated:
                        derivative *= shifts[quantum]
                gradient[axes[differentiated]] += derivative
        terms.append((state, coefficient, tuple(gradient)))
    return tuple(terms)


def _pair_input(
    component: ShellComponent,
    centers: tuple[int, int],
    prefixes: tuple[str, str],
    gradients: tuple[float, float],
    variables: Mapping[str, float],
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    """Expand two center labels into recurrence axes and shift inputs."""

    quantums = tuple((center, axis) for center in centers for axis in component[center])
    return (
        tuple(_AXIS_INDEX[axis] for _, axis in quantums),
        tuple(
            variables[f"{prefixes[centers.index(center)]}_{axis}"]
            for center, axis in quantums
        ),
        tuple(gradients[centers.index(center)] for center, _ in quantums),
    )


def evaluate_fused_shell_observables(
    spec: ShellClassSpec,
    component: Sequence[str],
    variables: Mapping[str, float],
    *,
    integral: IntegralIR | None = None,
) -> FusedShellResult:
    """Evaluate value and gradients using the fused recurrence execution shape.

    The numerical oracle follows the same derivative basis as the supplied
    mathematical IR.  In particular, an operator may recover any declared
    center from translation invariance; the historical ``A/B/C`` plus
    recovered ``D`` layout is only the default IntegralIR, not a property of
    the recurrence itself.
    """

    normalized = spec.validate_component(component)
    selected_integral = integral or build_integral_ir(spec)
    if selected_integral.spec != spec:
        raise ValueError("fused shell spec does not match its integral IR")
    first_scale = variables["first_product_scale"]
    second_scale = variables["second_product_scale"]
    third_scale = variables["third_product_scale"]
    # Older host callers supplied only the first three product scales.  The
    # fourth scale is the exact complement for the canonical pair, so retain
    # that compatibility while allowing explicit IRs to provide it directly.
    fourth_scale = variables.get("fourth_product_scale", 1.0 - third_scale)

    # Build one pair-term stream per independent derivative center.  Pair
    # coefficient derivatives depend on which center moves, while the value
    # recurrence and the Coulomb state table remain shared mathematically.
    zero_pair_gradient = (0.0, 0.0)
    pair_gradients = {
        0: ((first_scale - 1.0, first_scale), zero_pair_gradient),
        1: ((second_scale, second_scale - 1.0), zero_pair_gradient),
        2: (zero_pair_gradient, (third_scale - 1.0, third_scale)),
        3: (zero_pair_gradient, (fourth_scale, fourth_scale - 1.0)),
    }
    pair_terms_by_center = {}
    for center in selected_integral.independent_derivative_centers:
        first_gradient, second_gradient = pair_gradients[center]
        first_axes, first_shifts, first_shift_gradients = _pair_input(
            normalized,
            (0, 1),
            ("pa", "pb"),
            first_gradient,
            variables,
        )
        second_axes, second_shifts, second_shift_gradients = _pair_input(
            normalized,
            (2, 3),
            ("qc", "qd"),
            second_gradient,
            variables,
        )
        pair_terms_by_center[center] = (
            _pair_terms(
                first_axes,
                first_shifts,
                first_shift_gradients,
                variables["inverse_two_p"],
            ),
            _pair_terms(
                second_axes,
                second_shifts,
                second_shift_gradients,
                variables["inverse_two_q"],
            ),
        )

    # The value stream has no coefficient derivative and is independent of
    # which derivative centers the consumer requested.
    value_first_axes, value_first_shifts, _ = _pair_input(
        normalized,
        (0, 1),
        ("pa", "pb"),
        zero_pair_gradient,
        variables,
    )
    value_second_axes, value_second_shifts, _ = _pair_input(
        normalized,
        (2, 3),
        ("qc", "qd"),
        zero_pair_gradient,
        variables,
    )
    value_first_terms = _pair_terms(
        value_first_axes,
        value_first_shifts,
        (0.0,) * len(value_first_shifts),
        variables["inverse_two_p"],
    )
    value_second_terms = _pair_terms(
        value_second_axes,
        value_second_shifts,
        (0.0,) * len(value_second_shifts),
        variables["inverse_two_q"],
    )

    plan = build_fused_shell_plan(spec, integral=selected_integral)
    coulomb = {state: _coulomb_value(state, variables) for state in plan.coulomb_states}
    value = 0.0
    for first_state, first_coefficient, _ in value_first_terms:
        for second_state, second_coefficient, _ in value_second_terms:
            sign = -1.0 if sum(second_state) % 2 else 1.0
            state = tuple(first_state[axis] + second_state[axis] for axis in range(3))
            value += sign * first_coefficient * second_coefficient * coulomb[state]

    difference_scales = {
        0: first_scale,
        1: second_scale,
        2: -third_scale,
        3: -fourth_scale,
    }
    value_gradients = {
        center: [0.0, 0.0, 0.0]
        for center in selected_integral.independent_derivative_centers
    }
    for center, (first_terms, second_terms) in pair_terms_by_center.items():
        difference_scale = difference_scales[center]
        for first_state, first_coefficient, first_gradient in first_terms:
            for second_state, second_coefficient, second_gradient in second_terms:
                sign = -1.0 if sum(second_state) % 2 else 1.0
                state = tuple(
                    first_state[axis] + second_state[axis] for axis in range(3)
                )
                state_value = coulomb[state]
                coefficient = sign * first_coefficient * second_coefficient
                for coordinate in range(3):
                    derivative_state = list(state)
                    derivative_state[coordinate] += 1
                    scaled_derivative = (
                        coefficient
                        * difference_scale
                        * coulomb[tuple(derivative_state)]
                    )
                    coefficient_gradient = sign * (
                        first_gradient[coordinate] * second_coefficient
                        + first_coefficient * second_gradient[coordinate]
                    )
                    value_gradients[center][coordinate] += (
                        coefficient_gradient * state_value + scaled_derivative
                    )

    prefactor = variables["prefactor"]
    gradients_by_center = {
        center: tuple(
            prefactor
            * (
                value_gradients[center][coordinate]
                + value
                * variables[
                    f"decay_{('first', 'second', 'third', 'fourth')[center]}_{AXES[coordinate]}"
                ]
            )
            for coordinate in range(3)
        )
        for center in selected_integral.independent_derivative_centers
    }
    for center in selected_integral.recovered_derivative_centers:
        gradients_by_center[center] = tuple(
            -sum(
                gradients_by_center[independent][axis]
                for independent in selected_integral.independent_derivative_centers
            )
            for axis in range(3)
        )
    requested_centers = set(selected_integral.requested_derivative_centers)
    gradients = tuple(
        gradients_by_center.get(center, (0.0, 0.0, 0.0))
        if center in requested_centers
        else (0.0, 0.0, 0.0)
        for center in range(len(spec.angular))
    )
    return FusedShellResult(value=prefactor * value, gradients=gradients)


def evaluate_fused_shell_component(
    spec: ShellClassSpec,
    component: Sequence[str],
    variables: Mapping[str, float],
    *,
    integral: IntegralIR | None = None,
) -> tuple[tuple[float, float, float], ...]:
    """Compatibility view returning only all-center gradients."""

    return evaluate_fused_shell_observables(
        spec,
        component,
        variables,
        integral=integral,
    ).gradients


def evaluate_fused_shell_value(
    spec: ShellClassSpec,
    component: Sequence[str],
    variables: Mapping[str, float],
    *,
    integral: IntegralIR | None = None,
) -> float:
    """Return the ERI value consumed by generated Fock kernels."""

    return evaluate_fused_shell_observables(
        spec,
        component,
        variables,
        integral=integral,
    ).value
