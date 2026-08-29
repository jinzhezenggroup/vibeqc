"""Symbolic shell-class kernel builders.

The compact ``(p s|s s)`` pilot validates the compiler stages. The ``dppp``
builder mirrors VIBEQC's exact subset/Wick Gaussian-product expansion for one
Cartesian component, so the same DAG, differentiation, and CSE machinery can
be evaluated on the first profile-selected production shell class.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from .cuda import CudaEmitter
from .expr import Expr, Graph
from .shell_spec import AXES, DPPP_SPEC, ShellClassSpec

CENTERS = ("first", "second", "third", "fourth")
# Compatibility alias for the component-level dppp inspection CLI.  The
# ordering itself is generated from the declarative shell specification.
D_COMPONENTS = DPPP_SPEC.center_components[0]


@dataclass(frozen=True, slots=True)
class PsssKernel:
    """One symbolic ``(p_axis s|s s)`` value and all-center gradients."""

    graph: Graph
    p_axis: str
    variables: Mapping[str, Expr]
    boys_argument: Expr
    value: Expr
    gradients: tuple[tuple[Expr, Expr, Expr], ...]


@dataclass(frozen=True, slots=True)
class DpppComponentKernel:
    """One Cartesian component of canonical ``(d p|p p)`` and its gradient."""

    graph: Graph
    d_component: str
    p_components: tuple[str, str, str]
    variables: Mapping[str, Expr]
    boys_argument: Expr
    value: Expr
    gradients: tuple[tuple[Expr, Expr, Expr], ...]


@dataclass(frozen=True, slots=True)
class DpppContractionKernel:
    """Geometry-factored ``dppp`` component lowered for cooperative execution."""

    graph: Graph
    d_component: str
    p_components: tuple[str, str, str]
    variables: Mapping[str, Expr]
    value: Expr
    gradients: tuple[tuple[Expr, Expr, Expr], ...]


@dataclass(frozen=True, slots=True)
class ShellClassComponentKernel:
    """One component and its build-time symbolic all-center gradients."""

    graph: Graph
    spec: ShellClassSpec
    component: tuple[str, str, str, str]
    variables: Mapping[str, Expr]
    boys_argument: Expr
    value: Expr
    gradients: tuple[tuple[Expr, Expr, Expr], ...]


@dataclass(frozen=True, slots=True)
class ShellClassContractionKernel:
    """One shell component lowered around shared primitive geometry."""

    graph: Graph
    spec: ShellClassSpec
    component: tuple[str, str, str, str]
    variables: Mapping[str, Expr]
    value: Expr
    gradients: tuple[tuple[Expr, Expr, Expr], ...]


@dataclass(frozen=True, slots=True)
class WeightedShellContractionKernel:
    """Shell-wide value/gradient DAG after component weights are applied.

    All Cartesian components are cloned into one interned graph.  Variables
    with the same geometry name and structurally identical recurrence nodes
    therefore share one CUDA temporary, allowing low-order packed schedules to
    perform cross-component CSE before primitive execution.
    """

    graph: Graph
    spec: ShellClassSpec
    component_weights: tuple[Expr, ...]
    value: Expr
    gradients: tuple[tuple[Expr, Expr, Expr], ...]


@dataclass(frozen=True, slots=True)
class PackedForceGeometryAlgebra:
    """Backend-neutral scalar geometry consumed by packed force kernels."""

    graph: Graph
    variables: Mapping[str, Expr]
    rho: Expr
    inverse_two_p: Expr
    inverse_two_q: Expr
    pair_shifts: tuple[tuple[Expr, Expr, Expr], ...]
    difference: tuple[Expr, Expr, Expr]
    decay_gradients: tuple[tuple[Expr, Expr, Expr], ...]
    argument_squared_distance: Expr
    boys_argument: Expr
    prefactor: Expr
    primitive_coefficient: Expr

    @property
    def roots(self) -> tuple[Expr, ...]:
        """Return every computed scalar in deterministic storage order."""

        return self.roots_for_pair_shift_rows(4)

    def roots_for_pair_shift_rows(self, pair_shift_rows: int) -> tuple[Expr, ...]:
        """Return roots matching the packed record's stored shift rows."""

        if pair_shift_rows not in (3, 4):
            raise ValueError("packed geometry requires three or four shift rows")
        return (
            self.rho,
            self.inverse_two_p,
            self.inverse_two_q,
            *(item for center in self.pair_shifts[:pair_shift_rows] for item in center),
            *self.difference,
            *(item for center in self.decay_gradients for item in center),
            self.argument_squared_distance,
            self.boys_argument,
            self.prefactor,
            self.primitive_coefficient,
        )


def _squared_distance(
    graph: Graph,
    coordinates: Mapping[str, Mapping[str, Expr]],
    first: str,
    second: str,
) -> Expr:
    return graph.sum(
        (coordinates[first][axis] - coordinates[second][axis]).pow(2.0)
        for axis in AXES
    )


def build_packed_force_geometry_algebra() -> PackedForceGeometryAlgebra:
    """Describe packed primitive geometry without CUDA execution concepts."""

    graph = Graph()
    p = graph.variable("p")
    q = graph.variable("q")
    first_reduced_exponent = graph.variable("first_reduced_exponent")
    second_reduced_exponent = graph.variable("second_reduced_exponent")
    first_weighted_coefficient = graph.variable("first_weighted_coefficient")
    second_weighted_coefficient = graph.variable("second_weighted_coefficient")
    coordinates = {
        center: tuple(
            graph.variable(f"{center}_coordinate_{axis}") for axis in AXES
        )
        for center in CENTERS
    }
    product_p = tuple(graph.variable(f"product_p_{axis}") for axis in AXES)
    product_q = tuple(graph.variable(f"product_q_{axis}") for axis in AXES)
    variables = {
        "p": p,
        "q": q,
        "first_reduced_exponent": first_reduced_exponent,
        "second_reduced_exponent": second_reduced_exponent,
        "first_weighted_coefficient": first_weighted_coefficient,
        "second_weighted_coefficient": second_weighted_coefficient,
        **{
            f"{center}_coordinate_{axis}": coordinates[center][axis_index]
            for center in CENTERS
            for axis_index, axis in enumerate(AXES)
        },
        **{f"product_p_{axis}": product_p[index] for index, axis in enumerate(AXES)},
        **{f"product_q_{axis}": product_q[index] for index, axis in enumerate(AXES)},
    }

    rho = p * q / (p + q)
    inverse_two_p = 0.5 / p
    inverse_two_q = 0.5 / q
    pair_shifts = tuple(
        tuple(
            (product_p if center_index < 2 else product_q)[axis_index]
            - coordinates[center][axis_index]
            for axis_index in range(3)
        )
        for center_index, center in enumerate(CENTERS)
    )
    difference = tuple(product_p[index] - product_q[index] for index in range(3))
    first_separation = tuple(
        coordinates["first"][index] - coordinates["second"][index]
        for index in range(3)
    )
    second_separation = tuple(
        coordinates["third"][index] - coordinates["fourth"][index]
        for index in range(3)
    )
    decay_gradients = (
        tuple(-2 * first_reduced_exponent * item for item in first_separation),
        tuple(2 * first_reduced_exponent * item for item in first_separation),
        tuple(-2 * second_reduced_exponent * item for item in second_separation),
    )
    argument_squared_distance = graph.sum(item.pow(2) for item in difference)
    boys_argument = rho * argument_squared_distance
    prefactor = 34.986836655249725 / (p * q * (p + q).pow(0.5))
    primitive_coefficient = (
        first_weighted_coefficient * second_weighted_coefficient
    )
    return PackedForceGeometryAlgebra(
        graph=graph,
        variables=variables,
        rho=rho,
        inverse_two_p=inverse_two_p,
        inverse_two_q=inverse_two_q,
        pair_shifts=pair_shifts,
        difference=difference,
        decay_gradients=decay_gradients,
        argument_squared_distance=argument_squared_distance,
        boys_argument=boys_argument,
        prefactor=prefactor,
        primitive_coefficient=primitive_coefficient,
    )


def _integer_power(graph: Graph, value: Expr, exponent: int) -> Expr:
    """Build a small integer power from multiplies instead of runtime ``pow``."""

    if exponent < 0:
        raise ValueError("integer DAG powers must be non-negative")
    result = graph.constant(1.0)
    for _ in range(exponent):
        result = result * value
    return result


def _wick_matchings(
    indices: tuple[int, ...], axes: Sequence[int]
) -> Iterator[tuple[tuple[int, int], ...]]:
    """Enumerate every disjoint same-axis Wick matching exactly once."""

    if not indices:
        yield ()
        return
    first = indices[0]
    rest = indices[1:]
    # Leave the first quantum as a center-shift factor.
    yield from _wick_matchings(rest, axes)
    # Or contract it with one later quantum on the same Cartesian axis.
    for position, second in enumerate(rest):
        if axes[first] != axes[second]:
            continue
        remaining = rest[:position] + rest[position + 1 :]
        for tail in _wick_matchings(remaining, axes):
            yield ((first, second),) + tail


def _component_quantums(
    first_component: str,
    second_component: str,
    first_shifts: Mapping[str, Expr],
    second_shifts: Mapping[str, Expr],
) -> tuple[tuple[int, Expr], ...]:
    """Expand two Cartesian component labels into axis-major shift quanta."""

    quantums: list[tuple[int, Expr]] = []
    for axis_index, axis in enumerate(AXES):
        quantums.extend(
            (axis_index, first_shifts[axis])
            for _ in range(first_component.count(axis))
        )
        quantums.extend(
            (axis_index, second_shifts[axis])
            for _ in range(second_component.count(axis))
        )
    return tuple(quantums)


def _pair_expansion(
    graph: Graph,
    quantums: Sequence[tuple[int, Expr]],
    inverse_two_exponent: Expr,
) -> tuple[tuple[tuple[int, int, int], Expr], ...]:
    """Generate exact Hermite derivative states and subset/Wick coefficients."""

    axes = tuple(axis for axis, _ in quantums)
    shifts = tuple(shift for _, shift in quantums)
    terms = []
    for subset in range(1 << len(quantums)):
        selected = tuple(
            quantum
            for quantum in range(len(quantums))
            if subset & (1 << quantum)
        )
        selected_set = frozenset(selected)
        derivative_orders = tuple(
            sum(axes[quantum] == axis for quantum in selected)
            for axis in range(3)
        )
        remaining = tuple(
            quantum
            for quantum in range(len(quantums))
            if quantum not in selected_set
        )
        coefficient_terms = []
        for matching in _wick_matchings(remaining, axes):
            removed = frozenset(quantum for pair in matching for quantum in pair)
            coefficient = _integer_power(
                graph,
                inverse_two_exponent,
                len(selected) + len(matching),
            )
            for quantum in remaining:
                if quantum not in removed:
                    coefficient = coefficient * shifts[quantum]
            coefficient_terms.append(coefficient)
        terms.append((derivative_orders, graph.sum(coefficient_terms)))
    return tuple(terms)


def _axis_wick_multiplicity(order: int, pairs: int) -> int:
    """Match the closed Cartesian Coulomb derivative multiplicity."""

    numerator = 1
    for value in range(order - 2 * pairs + 1, order + 1):
        numerator *= value
    denominator = (2**pairs)
    for value in range(2, pairs + 1):
        denominator *= value
    return numerator // denominator


def _coulomb_derivative(
    graph: Graph,
    derivative_orders: tuple[int, int, int],
    rho: Expr,
    difference: Mapping[str, Expr],
    boys: Sequence[Expr],
) -> Expr:
    """Build one exact Cartesian Coulomb derivative through total order six."""

    total_order = sum(derivative_orders)
    negative_two_rho = -2.0 * rho
    terms = []
    for x_pairs in range(derivative_orders[0] // 2 + 1):
        for y_pairs in range(derivative_orders[1] // 2 + 1):
            for z_pairs in range(derivative_orders[2] // 2 + 1):
                contraction_count = x_pairs + y_pairs + z_pairs
                boys_order = total_order - contraction_count
                multiplicity = (
                    _axis_wick_multiplicity(derivative_orders[0], x_pairs)
                    * _axis_wick_multiplicity(derivative_orders[1], y_pairs)
                    * _axis_wick_multiplicity(derivative_orders[2], z_pairs)
                )
                terms.append(
                    float(multiplicity)
                    * _integer_power(graph, negative_two_rho, boys_order)
                    * _integer_power(
                        graph, difference["x"], derivative_orders[0] - 2 * x_pairs
                    )
                    * _integer_power(
                        graph, difference["y"], derivative_orders[1] - 2 * y_pairs
                    )
                    * _integer_power(
                        graph, difference["z"], derivative_orders[2] - 2 * z_pairs
                    )
                    * boys[boys_order]
                )
    return graph.sum(terms)


def _validated_dppp_components(
    d_component: str, p_components: Sequence[str]
) -> tuple[str, str, str]:
    """Validate and normalize one Cartesian component of canonical `dppp`."""

    normalized = tuple(p_components)
    DPPP_SPEC.validate_component((d_component, *normalized))
    return normalized


def _shell_component_value(
    graph: Graph,
    component: tuple[str, str, str, str],
    shifts: Mapping[str, Mapping[str, Expr]],
    inverse_two_p: Expr,
    inverse_two_q: Expr,
    rho: Expr,
    difference: Mapping[str, Expr],
    boys: Sequence[Expr],
) -> Expr:
    """Build exact pair/Coulomb algebra for any four-center component."""

    first_expansion = _pair_expansion(
        graph,
        _component_quantums(
            component[0], component[1], shifts["pa"], shifts["pb"]
        ),
        inverse_two_p,
    )
    second_expansion = _pair_expansion(
        graph,
        _component_quantums(
            component[2], component[3], shifts["qc"], shifts["qd"]
        ),
        inverse_two_q,
    )
    coulomb_cache: dict[tuple[int, int, int], Expr] = {}

    def coulomb(orders: tuple[int, int, int]) -> Expr:
        expression = coulomb_cache.get(orders)
        if expression is None:
            expression = _coulomb_derivative(
                graph, orders, rho, difference, boys
            )
            coulomb_cache[orders] = expression
        return expression

    terms = []
    for first_orders, first_coefficient in first_expansion:
        for second_orders, second_coefficient in second_expansion:
            orders = tuple(
                first_orders[axis] + second_orders[axis]
                for axis in range(3)
            )
            sign = -1.0 if sum(second_orders) % 2 else 1.0
            terms.append(
                sign
                * first_coefficient
                * second_coefficient
                * coulomb(orders)
            )
    return graph.sum(terms)


def build_psss_kernel(p_axis: str) -> PsssKernel:
    """Build the canonical primitive value and symbolic nuclear derivatives."""

    if p_axis not in AXES:
        raise ValueError(f"unsupported p axis {p_axis!r}")
    graph = Graph()
    alpha = graph.variable("alpha")
    beta = graph.variable("beta")
    gamma = graph.variable("gamma")
    delta = graph.variable("delta")
    coordinates = {
        center: {
            axis: graph.variable(f"{center}_{axis}") for axis in AXES
        }
        for center in CENTERS
    }
    variables: dict[str, Expr] = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
    }
    for center in CENTERS:
        variables.update(
            {
                f"{center}_{axis}": coordinates[center][axis]
                for axis in AXES
            }
        )

    p = alpha + beta
    q = gamma + delta
    mu = alpha * beta / p
    nu = gamma * delta / q
    rho = p * q / (p + q)
    product_p = {
        axis: (
            alpha * coordinates["first"][axis]
            + beta * coordinates["second"][axis]
        )
        / p
        for axis in AXES
    }
    product_q = {
        axis: (
            gamma * coordinates["third"][axis]
            + delta * coordinates["fourth"][axis]
        )
        / q
        for axis in AXES
    }
    product_difference = {
        axis: product_p[axis] - product_q[axis] for axis in AXES
    }
    boys_argument = rho * graph.sum(
        product_difference[axis].pow(2.0) for axis in AXES
    )
    boys = tuple(graph.variable(f"boys_{order}") for order in range(3))
    pair_decay = graph.exponential(
        -mu * _squared_distance(graph, coordinates, "first", "second")
        - nu * _squared_distance(graph, coordinates, "third", "fourth")
    )
    pi = graph.variable("kPi")
    normalization = (
        2.0 * pi.pow(2.5) / (p * q * (p + q).pow(0.5))
    )
    pa = product_p[p_axis] - coordinates["first"][p_axis]
    primitive_value = (
        pa * boys[0] - (rho / p) * product_difference[p_axis] * boys[1]
    )
    value = normalization * pair_decay * primitive_value

    independent_gradients: list[tuple[Expr, Expr, Expr]] = []
    for center in CENTERS[:3]:
        center_gradients: list[Expr] = []
        for axis in AXES:
            variable = coordinates[center][axis]
            argument_derivative = graph.differentiate(boys_argument, variable)
            leaf_derivatives = {
                "boys_0": -boys[1] * argument_derivative,
                "boys_1": -boys[2] * argument_derivative,
            }
            center_gradients.append(
                graph.differentiate(value, variable, leaf_derivatives)
            )
        independent_gradients.append(tuple(center_gradients))
    fourth_gradient = tuple(
        -graph.sum(independent_gradients[center][axis] for center in range(3))
        for axis in range(3)
    )
    gradients = tuple(independent_gradients) + (fourth_gradient,)
    return PsssKernel(
        graph=graph,
        p_axis=p_axis,
        variables=variables,
        boys_argument=boys_argument,
        value=value,
        gradients=gradients,
    )


def build_shell_class_component_kernel(
    spec: ShellClassSpec,
    component: Sequence[str],
) -> ShellClassComponentKernel:
    """Build one shell component and its compile-time symbolic derivatives.

    The value expression follows the same subset/Wick pair expansion and
    closed Cartesian Coulomb derivatives as the production oracle.  Automatic
    differentiation runs only in Python; emitted CUDA remains scalar analytic
    code with no runtime AD types or tape.
    """

    normalized = spec.validate_component(component)
    maximum_order = spec.maximum_force_coulomb_order

    graph = Graph()
    alpha = graph.variable("alpha")
    beta = graph.variable("beta")
    gamma = graph.variable("gamma")
    delta = graph.variable("delta")
    coordinates = {
        center: {
            axis: graph.variable(f"{center}_{axis}") for axis in AXES
        }
        for center in CENTERS
    }
    variables: dict[str, Expr] = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
    }
    for center in CENTERS:
        variables.update(
            {
                f"{center}_{axis}": coordinates[center][axis]
                for axis in AXES
            }
        )

    p = alpha + beta
    q = gamma + delta
    mu = alpha * beta / p
    nu = gamma * delta / q
    rho = p * q / (p + q)
    product_p = {
        axis: (
            alpha * coordinates["first"][axis]
            + beta * coordinates["second"][axis]
        )
        / p
        for axis in AXES
    }
    product_q = {
        axis: (
            gamma * coordinates["third"][axis]
            + delta * coordinates["fourth"][axis]
        )
        / q
        for axis in AXES
    }
    difference = {
        axis: product_p[axis] - product_q[axis] for axis in AXES
    }
    boys_argument = rho * graph.sum(
        difference[axis].pow(2.0) for axis in AXES
    )
    boys = tuple(
        graph.variable(f"boys_{order}") for order in range(maximum_order + 1)
    )
    shifts = {
        "pa": {
            axis: product_p[axis] - coordinates["first"][axis] for axis in AXES
        },
        "pb": {
            axis: product_p[axis] - coordinates["second"][axis] for axis in AXES
        },
        "qc": {
            axis: product_q[axis] - coordinates["third"][axis] for axis in AXES
        },
        "qd": {
            axis: product_q[axis] - coordinates["fourth"][axis] for axis in AXES
        },
    }
    primitive_value = _shell_component_value(
        graph,
        normalized,
        shifts,
        0.5 / p,
        0.5 / q,
        rho,
        difference,
        boys,
    )
    pair_decay = graph.exponential(
        -mu * _squared_distance(graph, coordinates, "first", "second")
        - nu * _squared_distance(graph, coordinates, "third", "fourth")
    )
    pi = graph.variable("kPi")
    normalization = 2.0 * pi.pow(2.5) / (p * q * (p + q).pow(0.5))
    value = normalization * pair_decay * primitive_value

    independent_gradients: list[tuple[Expr, Expr, Expr]] = []
    for center in CENTERS[:3]:
        center_gradients = []
        for axis in AXES:
            variable = coordinates[center][axis]
            argument_derivative = graph.differentiate(boys_argument, variable)
            leaf_derivatives = {
                f"boys_{order}": -boys[order + 1] * argument_derivative
                for order in range(maximum_order)
            }
            center_gradients.append(
                graph.differentiate(value, variable, leaf_derivatives)
            )
        independent_gradients.append(tuple(center_gradients))
    fourth_gradient = tuple(
        -graph.sum(independent_gradients[center][axis] for center in range(3))
        for axis in range(3)
    )
    gradients = tuple(independent_gradients) + (fourth_gradient,)
    return ShellClassComponentKernel(
        graph=graph,
        spec=spec,
        component=normalized,
        variables=variables,
        boys_argument=boys_argument,
        value=value,
        gradients=gradients,
    )


def build_dppp_component_kernel(
    d_component: str,
    p_components: Sequence[str],
) -> DpppComponentKernel:
    """Build one canonical ``(d p|p p)`` component via the generic compiler."""

    normalized = _validated_dppp_components(d_component, p_components)
    kernel = build_shell_class_component_kernel(
        DPPP_SPEC, (d_component, *normalized)
    )
    return DpppComponentKernel(
        graph=kernel.graph,
        d_component=d_component,
        p_components=normalized,
        variables=kernel.variables,
        boys_argument=kernel.boys_argument,
        value=kernel.value,
        gradients=kernel.gradients,
    )


def build_shell_class_contraction_kernel(
    spec: ShellClassSpec,
    component: Sequence[str],
) -> ShellClassContractionKernel:
    """Lower one shell component around shared primitive-shell geometry.

    A cooperative shell-quartet worker can compute product centers, Boys
    values, the common prefactor, and decay derivatives once. This DAG then
    contains only component-specific pair/Coulomb algebra and uses symbolic AD
    with respect to shared shifts and the product-center difference.
    """

    normalized = spec.validate_component(component)
    maximum_order = spec.maximum_force_coulomb_order

    graph = Graph()
    inverse_two_p = graph.variable("inverse_two_p")
    inverse_two_q = graph.variable("inverse_two_q")
    rho = graph.variable("rho")
    difference = {
        axis: graph.variable(f"difference_{axis}") for axis in AXES
    }
    shifts = {
        prefix: {
            axis: graph.variable(f"{prefix}_{axis}") for axis in AXES
        }
        for prefix in ("pa", "pb", "qc", "qd")
    }
    boys = tuple(
        graph.variable(f"boys_{order}") for order in range(maximum_order + 1)
    )
    first_product_scale = graph.variable("first_product_scale")
    second_product_scale = graph.variable("second_product_scale")
    third_product_scale = graph.variable("third_product_scale")
    prefactor = graph.variable("prefactor")
    decay_gradients = {
        center: {
            axis: graph.variable(f"decay_{center}_{axis}") for axis in AXES
        }
        for center in CENTERS[:3]
    }
    variables: dict[str, Expr] = {
        "inverse_two_p": inverse_two_p,
        "inverse_two_q": inverse_two_q,
        "rho": rho,
        "first_product_scale": first_product_scale,
        "second_product_scale": second_product_scale,
        "third_product_scale": third_product_scale,
        "prefactor": prefactor,
    }
    variables.update({f"difference_{axis}": difference[axis] for axis in AXES})
    for prefix in shifts:
        variables.update(
            {f"{prefix}_{axis}": shifts[prefix][axis] for axis in AXES}
        )
    for center in CENTERS[:3]:
        variables.update(
            {
                f"decay_{center}_{axis}": decay_gradients[center][axis]
                for axis in AXES
            }
        )

    value = _shell_component_value(
        graph,
        normalized,
        shifts,
        inverse_two_p,
        inverse_two_q,
        rho,
        difference,
        boys,
    )

    independent_gradients: list[tuple[Expr, Expr, Expr]] = []
    for center_index, center in enumerate(CENTERS[:3]):
        center_gradients = []
        for axis in AXES:
            argument_derivative = 2.0 * rho * difference[axis]
            leaf_derivatives = {
                f"boys_{order}": -boys[order + 1] * argument_derivative
                for order in range(maximum_order)
            }
            difference_gradient = graph.differentiate(
                value, difference[axis], leaf_derivatives
            )
            if center_index < 2:
                pair_gradient = (
                    (first_product_scale - 1.0)
                    * graph.differentiate(value, shifts["pa"][axis])
                    + first_product_scale
                    * graph.differentiate(value, shifts["pb"][axis])
                )
                unscaled = (
                    pair_gradient
                    + first_product_scale * difference_gradient
                    if center_index == 0
                    else -pair_gradient
                    + second_product_scale * difference_gradient
                )
            else:
                pair_gradient = (
                    (third_product_scale - 1.0)
                    * graph.differentiate(value, shifts["qc"][axis])
                    + third_product_scale
                    * graph.differentiate(value, shifts["qd"][axis])
                )
                unscaled = pair_gradient - third_product_scale * difference_gradient
            center_gradients.append(
                prefactor
                * (unscaled + value * decay_gradients[center][axis])
            )
        independent_gradients.append(tuple(center_gradients))
    fourth_gradient = tuple(
        -graph.sum(independent_gradients[center][axis] for center in range(3))
        for axis in range(3)
    )
    gradients = tuple(independent_gradients) + (fourth_gradient,)
    return ShellClassContractionKernel(
        graph=graph,
        spec=spec,
        component=normalized,
        variables=variables,
        value=value,
        gradients=gradients,
    )


def _clone_expression(
    expression: Expr,
    target: Graph,
    memo: dict[int, Expr],
) -> Expr:
    """Clone one DAG root while interning shared variables and operations."""

    cached = memo.get(expression.identifier)
    if cached is not None:
        return cached
    node = expression.graph.node(expression)
    if node.operation == "constant":
        cloned = target.clone_constant(node)
    elif node.operation == "variable":
        cloned = target.variable(str(node.payload))
    else:
        arguments = tuple(
            _clone_expression(
                Expr(expression.graph, identifier),
                target,
                memo,
            )
            for identifier in node.arguments
        )
        if node.operation == "add":
            cloned = (
                target.add(arguments[0], arguments[1])
                if len(arguments) == 2
                else target.add_many(arguments)
            )
        elif node.operation == "multiply":
            cloned = (
                target.multiply(arguments[0], arguments[1])
                if len(arguments) == 2
                else target.multiply_many(arguments)
            )
        elif node.operation == "reciprocal":
            cloned = target.reciprocal(arguments[0])
        elif node.operation == "exp":
            cloned = target.exponential(arguments[0])
        elif node.operation == "power":
            cloned = target.power(arguments[0], float(node.payload))
        else:
            raise ValueError(f"unsupported cloned operation {node.operation!r}")
    memo[expression.identifier] = cloned
    return cloned


def build_weighted_shell_contraction_kernel(
    spec: ShellClassSpec,
    component_indices: Sequence[int] | None = None,
) -> WeightedShellContractionKernel:
    """Build one density-weightable DAG spanning every shell component.

    The component kernels intentionally originate from the existing symbolic
    oracle.  Cloning them into a shared graph preserves that correctness source
    while exposing horizontal CSE that is invisible to one-component-at-a-time
    CUDA lowering.  ``component_indices`` optionally bounds CSE to a stable
    subset so register-sensitive emitters can trade a small amount of
    recomputation for shorter live ranges without changing the mathematics.
    """

    selected_components = (
        tuple(range(spec.component_count))
        if component_indices is None
        else tuple(component_indices)
    )
    if not selected_components:
        raise ValueError("weighted shell contraction requires a component")
    if len(set(selected_components)) != len(selected_components):
        raise ValueError("weighted shell component indices must be unique")
    if any(
        component < 0 or component >= spec.component_count
        for component in selected_components
    ):
        raise ValueError("weighted shell component index is out of range")

    graph = Graph()
    component_weights = tuple(
        graph.variable(f"component_weight_{component}")
        for component in selected_components
    )
    weighted_values = []
    weighted_gradients: list[list[list[Expr]]] = [
        [[] for _ in AXES] for _ in CENTERS
    ]
    for component_index, weight in zip(
        selected_components,
        component_weights,
        strict=True,
    ):
        component = spec.components[component_index]
        kernel = build_shell_class_contraction_kernel(spec, component)
        memo: dict[int, Expr] = {}
        weighted_values.append(
            weight * _clone_expression(kernel.value, graph, memo)
        )
        for center in range(4):
            for axis in range(3):
                weighted_gradients[center][axis].append(
                    weight
                    * _clone_expression(
                        kernel.gradients[center][axis],
                        graph,
                        memo,
                    )
                )
    prefactor = graph.variable("prefactor")
    value = prefactor * graph.sum(weighted_values)
    gradients = tuple(
        tuple(graph.sum(weighted_gradients[center][axis]) for axis in range(3))
        for center in range(4)
    )
    return WeightedShellContractionKernel(
        graph=graph,
        spec=spec,
        component_weights=component_weights,
        value=value,
        gradients=gradients,
    )


def build_dppp_contraction_kernel(
    d_component: str,
    p_components: Sequence[str],
) -> DpppContractionKernel:
    """Lower one ``dppp`` component via the generic shell-class compiler."""

    normalized = _validated_dppp_components(d_component, p_components)
    kernel = build_shell_class_contraction_kernel(
        DPPP_SPEC, (d_component, *normalized)
    )
    return DpppContractionKernel(
        graph=kernel.graph,
        d_component=d_component,
        p_components=normalized,
        variables=kernel.variables,
        value=kernel.value,
        gradients=kernel.gradients,
    )


def emit_psss_cuda(kernel: PsssKernel) -> str:
    """Emit one AOT-ready CUDA device function for the pilot shell class."""

    variable_code = {
        "alpha": "alpha",
        "beta": "beta",
        "gamma": "gamma",
        "delta": "delta",
        "kPi": "kPi",
        "boys_0": "boys[0]",
        "boys_1": "boys[1]",
        "boys_2": "boys[2]",
    }
    for center in CENTERS:
        for axis in AXES:
            variable_code[f"{center}_{axis}"] = f"{center}.{axis}"
    emitter = CudaEmitter(kernel.graph, variable_code)
    emitter.emit([kernel.boys_argument])
    argument_reference = emitter.reference(kernel.boys_argument)
    emitter.lines.append("  double boys[3];")
    emitter.lines.append(
        f"  boys_values<2>({argument_reference}, boys);"
    )
    roots = [item for center in kernel.gradients for item in center]
    emitter.emit(roots)

    lines = [
        (
            "/** Generated symbolic/CSE derivative for canonical "
            f"(p{kernel.p_axis} s|s s). */"
        ),
        f"__device__ void generated_psss_{kernel.p_axis}_gradient(",
        "    double alpha,",
        "    const Vec3<double>& first,",
        "    double beta,",
        "    const Vec3<double>& second,",
        "    double gamma,",
        "    const Vec3<double>& third,",
        "    double delta,",
        "    const Vec3<double>& fourth,",
        "    double (&gradient)[4][3]) {",
        *emitter.lines,
    ]
    for center, center_gradients in enumerate(kernel.gradients):
        for coordinate, expression in enumerate(center_gradients):
            lines.append(
                f"  gradient[{center}][{coordinate}] = "
                f"{emitter.reference(expression)};"
            )
    lines.append("}")
    return "\n".join(lines) + "\n"


def emit_dppp_component_cuda(kernel: DpppComponentKernel) -> str:
    """Emit scalar analytic CUDA for one generated ``dppp`` component."""

    variable_code = {
        "alpha": "alpha",
        "beta": "beta",
        "gamma": "gamma",
        "delta": "delta",
        "kPi": "kPi",
        **{f"boys_{order}": f"boys[{order}]" for order in range(7)},
    }
    for center in CENTERS:
        for axis in AXES:
            variable_code[f"{center}_{axis}"] = f"{center}.{axis}"
    emitter = CudaEmitter(kernel.graph, variable_code)
    emitter.emit([kernel.boys_argument])
    argument_reference = emitter.reference(kernel.boys_argument)
    emitter.lines.append("  double boys[7];")
    emitter.lines.append(f"  boys_values<6>({argument_reference}, boys);")
    roots = [item for center in kernel.gradients for item in center]
    emitter.emit(roots)

    p_label = "".join(kernel.p_components)
    function_name = f"generated_dppp_{kernel.d_component}_{p_label}_gradient"
    lines = [
        (
            "/** Generated symbolic/CSE derivative for canonical "
            f"({kernel.d_component} {kernel.p_components[0]}|"
            f"{kernel.p_components[1]} {kernel.p_components[2]}). */"
        ),
        f"__device__ void {function_name}(",
        "    double alpha,",
        "    const Vec3<double>& first,",
        "    double beta,",
        "    const Vec3<double>& second,",
        "    double gamma,",
        "    const Vec3<double>& third,",
        "    double delta,",
        "    const Vec3<double>& fourth,",
        "    double (&gradient)[4][3]) {",
        *emitter.lines,
    ]
    for center, center_gradients in enumerate(kernel.gradients):
        for coordinate, expression in enumerate(center_gradients):
            lines.append(
                f"  gradient[{center}][{coordinate}] = "
                f"{emitter.reference(expression)};"
            )
    lines.append("}")
    return "\n".join(lines) + "\n"


def emit_dppp_contraction_cuda(kernel: DpppContractionKernel) -> str:
    """Emit a component evaluator that consumes cooperative common geometry."""

    variable_code = {
        "inverse_two_p": "geometry.inverse_two_p",
        "inverse_two_q": "geometry.inverse_two_q",
        "rho": "geometry.rho",
        "first_product_scale": "geometry.product_scales[0]",
        "second_product_scale": "geometry.product_scales[1]",
        "third_product_scale": "geometry.product_scales[2]",
        "prefactor": "geometry.prefactor",
        **{
            f"difference_{axis}": f"geometry.difference.{axis}"
            for axis in AXES
        },
        **{f"boys_{order}": f"geometry.boys[{order}]" for order in range(7)},
    }
    for center, prefix in enumerate(("pa", "pb", "qc", "qd")):
        for axis in AXES:
            variable_code[f"{prefix}_{axis}"] = (
                f"geometry.pair_shifts[{center}].{axis}"
            )
    for center_index, center in enumerate(CENTERS[:3]):
        for axis_index, axis in enumerate(AXES):
            variable_code[f"decay_{center}_{axis}"] = (
                f"geometry.decay_gradients[{center_index}][{axis_index}]"
            )

    emitter = CudaEmitter(kernel.graph, variable_code)
    roots = [item for center in kernel.gradients for item in center]
    emitter.emit(roots)
    p_label = "".join(kernel.p_components)
    function_name = (
        f"generated_dppp_{kernel.d_component}_{p_label}_factored_gradient"
    )
    lines = [
        (
            "/** Component algebra consuming one cooperatively shared primitive "
            "geometry. */"
        ),
        f"__device__ void {function_name}(",
        "    const GeneratedDpppGeometry& geometry,",
        "    double (&gradient)[4][3]) {",
        *emitter.lines,
    ]
    for center, center_gradients in enumerate(kernel.gradients):
        for coordinate, expression in enumerate(center_gradients):
            lines.append(
                f"  gradient[{center}][{coordinate}] = "
                f"{emitter.reference(expression)};"
            )
    lines.append("}")
    return "\n".join(lines) + "\n"
