"""Consumer-directed shell-local second derivatives in the shared scalar DAG.

All differentiation happens during code generation. External weights, orbital
coefficients, exponents, nuclear charges, omega and HVP directions are fixed.
These partial integral derivatives do not include electronic response and do
not enable a molecular Hessian. Raw blocks lower one AO component at a time;
weighted outputs combine a bounded component subset before differentiation.
"""

from dataclasses import dataclass, replace
from functools import cache
from itertools import product
from math import sqrt

from .blocks import SecondDerivative, TensorLayout, WeightDescriptor, WeightedDerivative
from .expr import AlgebraForm, Expr, Graph
from .ir import FOUR_CENTER_ERI_OPERATOR, IntegralIR, OperatorFamily
from .one_electron_values import (
    build_one_electron_component_kernel,
    build_one_electron_value_ir,
)
from .second_order_layout import CenterRecovery, HessianLayout, second_center_recovery
from .shell_spec import AXES, cartesian_components
from .weighted_eri import CENTERS, build_weighted_eri_ir, build_weighted_eri_kernel


def require_second_consumer(integral):
    """Validate the single-consumer invariant before reading second-order fields.

    General IntegralIR can represent multiple consumers. Native second-order
    artifacts and their record/public-input adapters implement exactly one.
    This inexpensive check also protects manually constructed artifact objects
    without repeating native hash validation for every streamed primitive.
    """
    if integral.derivative is None or integral.derivative.order != 2:
        raise ValueError("second derivative lowering requires explicit order two")
    if len(integral.contractions) != 1 or not isinstance(
        integral.contractions[0], SecondDerivative
    ):
        raise ValueError(
            "second derivative lowering requires one explicit second-order consumer"
        )
    return integral.contractions[0]


def _second_ir(integral, output, packing, memory_budget_bytes):
    """Retain the value operator/signature and declare a separate output ABI."""
    derivative = integral.operator.nuclear_derivative(order=2)
    centers = derivative.requested_centers(integral.operator)
    layout = (
        TensorLayout(("center", "xyz"), (len(centers), 3))
        if output == "weighted_hvp"
        else HessianLayout(centers, packing).tensor_layout
    )
    weights = None
    if output == "raw_hessian":
        layout = TensorLayout(
            layout.indices + integral.signature.tensor_indices,
            layout.shape + integral.signature.component_shape,
        )
    else:
        weights = WeightDescriptor(
            "external_fixed_weight",
            TensorLayout(
                integral.signature.tensor_indices, integral.signature.component_shape
            ),
        )
    consumer = SecondDerivative(
        layout,
        memory_budget_bytes,
        weights,
        output,
        packing,
        "fixed_direction" if output == "weighted_hvp" else None,
    )
    return replace(integral, derivative=derivative, contractions=(consumer,))


def build_one_electron_second_ir(
    family,
    angular,
    *,
    charge=1.0,
    output="raw_hessian",
    packing="dense",
    memory_budget_bytes=4 * 1024**2,
):
    """Declare S/T/V partial second derivatives for public s/p/d/f shell pairs."""
    return _second_ir(
        build_one_electron_value_ir(family, angular, charge=charge),
        output,
        packing,
        memory_budget_bytes,
    )


def build_eri_second_ir(
    angular,
    *,
    operator=FOUR_CENTER_ERI_OPERATOR,
    output="weighted_hvp",
    packing="dense",
    memory_budget_bytes=4 * 1024**2,
):
    """Declare four-center raw, fixed-weight Hessian or directional outputs.

    Range semantics can be serialized, but executable support is checked by
    lowering independently. No DF/ECP second derivatives are inferred here.
    """
    return _second_ir(
        build_weighted_eri_ir(angular, operator=operator),
        output,
        packing,
        memory_budget_bytes,
    )


@dataclass(frozen=True)
class SecondDerivativeKernel:
    """One explicit AO subset and coordinate-output tile with shared roots.

    ``outputs`` follows ``output_indices`` in the flattened dense/svec/HVP
    coordinate layout. A raw kernel has exactly one selected AO component;
    its caller assembles that component into the declared AO tensor axes.
    The extra value/gradient roots serve diagnostics and can be pruned by a
    backend that only emits second-order outputs. HVP construction never builds
    a dense Hessian, even temporarily in the generation-time graph.
    """

    integral: IntegralIR
    component_indices: tuple[int, ...]
    graph: Graph
    value: Expr
    gradients: tuple[Expr, ...]
    outputs: tuple[Expr, ...]
    output_indices: tuple[int, ...]
    recovery: CenterRecovery
    boys_argument: Expr | None
    boys_count: int


def _eri_differentiate(graph, maximum_order):
    """Total coordinate response of the existing geometry-factored ERI DAG.

    The logarithmic Gaussian decay leaves introduced by the first derivative
    themselves vary with coordinates. Their diagonal slopes must be included
    in the second derivative, while pair scales and inverse exponents remain
    constant. Omitting these slopes breaks even the ssss Hessian.
    """

    @cache
    def responses(center, axis):
        scale = graph.variable(f"{CENTERS[center]}_product_scale")
        difference_scale = scale if center < 2 else -scale
        leaves = {
            "prefactor": graph.variable("prefactor")
            * graph.variable(f"decay_{CENTERS[center]}_{axis}"),
            f"difference_{axis}": difference_scale,
        }
        for slot, prefix in enumerate(("pa", "pb") if center < 2 else ("qc", "qd")):
            leaves[f"{prefix}_{axis}"] = scale - int(slot == center % 2)
        dt = (
            2
            * graph.variable("rho")
            * graph.variable(f"difference_{axis}")
            * difference_scale
        )
        leaves.update(
            {
                f"boys_{n}": -graph.variable(f"boys_{n + 1}") * dt
                for n in range(maximum_order)
            }
        )
        pair = center // 2 * 2
        slope = (
            graph.variable(f"{CENTERS[pair]}_product_scale")
            * graph.variable(f"{CENTERS[pair + 1]}_product_scale")
            / graph.variable("inverse_two_p" if center < 2 else "inverse_two_q")
        )
        for other in (pair, pair + 1):
            leaves[f"decay_{CENTERS[other]}_{axis}"] = (
                -slope if center == other else slope
            )
        return leaves

    def differentiate(root, center, axis):
        return graph.differentiate(
            root, graph.variable(f"coordinate_{center}_{axis}"), responses(center, axis)
        )

    return differentiate


def _one_electron_primal(integral, indices, primal_form):
    """Contract selected S/T/V components before AD, sharing pair/Boys states."""
    charge = (
        integral.operator.external_centers[0].charge
        if integral.operator.external_centers
        else 1.0
    )
    template = build_one_electron_value_ir(
        integral.operator.family, integral.signature.angular, charge=charge
    )
    value_ir = replace(integral, derivative=None, contractions=template.contractions)
    components = tuple(
        product(*(cartesian_components(order) for order in integral.signature.angular))
    )
    graph = Graph()
    values = []
    consumer = integral.contractions[0]
    for index in indices:
        kernel = build_one_electron_component_kernel(
            value_ir, components[index], graph=graph
        )
        weight = (
            graph.constant(1)
            if consumer.weights is None
            else graph.variable(f"component_weight_{index}")
        )
        values.append(weight * kernel.value)
    factor = consumer.output_sign
    if consumer.weights is not None:
        factor *= consumer.weights.sign * consumer.weights.prefactor
    value = factor * graph.sum(values)
    auxiliary = () if kernel.boys_argument is None else (kernel.boys_argument,)
    graph, roots = graph.apply_algebra_form((value, *auxiliary), primal_form)
    argument = roots[1] if auxiliary else None
    count = kernel.boys_count + (2 if argument is not None else 0)

    @cache
    def responses(center, axis):
        variable = graph.variable(f"{'abc'[center]}_{axis}")
        dt = None if argument is None else graph.differentiate(argument, variable)
        return variable, {
            f"boys_{n}": -graph.variable(f"boys_{n + 1}") * dt for n in range(count - 1)
        }

    def differentiate(root, center, axis):
        variable, leaves = responses(center, axis)
        return graph.differentiate(root, variable, leaves)

    return graph, roots[0], argument, count, differentiate


def build_second_derivative_kernel(
    integral: IntegralIR,
    component_indices=None,
    *,
    output_indices=None,
    primal_form=AlgebraForm.FACTORED_NARY,
    output_form=AlgebraForm.BINARY,
):
    """Lower one bounded second-order consumer without changing first-force ABI.

    Select one AO component for raw output, or one to 64 components for a fixed
    weighted output. ``output_indices`` selects only required coordinate pairs
    or HVP rows before AD, so a backend can bound liveness by output tiling.
    Algebra choices expose valid optimize-before/after-AD validation schedules.
    """
    consumer = require_second_consumer(integral)
    count = integral.signature.component_count
    indices = (
        tuple(range(count)) if component_indices is None else tuple(component_indices)
    )
    if (
        not 1 <= len(indices) <= 64
        or len(set(indices)) != len(indices)
        or any(type(i) is not int or not 0 <= i < count for i in indices)
    ):
        raise ValueError("select one to 64 distinct AO components within the shell")
    if consumer.output == "raw_hessian" and len(indices) != 1:
        raise ValueError(
            "raw second derivative lowering requires one explicit AO component"
        )
    if (
        consumer.weights is not None
        and consumer.weights.layout.shape != integral.signature.component_shape
    ):
        raise ValueError(
            "lowering requires the full shell weight layout and an explicit component subset"
        )
    primal_form, output_form = AlgebraForm(primal_form), AlgebraForm(output_form)
    recovery = second_center_recovery(integral.operator, integral.derivative)
    dimension = 3 * len(recovery.centers)
    hvp = consumer.output == "weighted_hvp"
    pairs = HessianLayout(recovery.centers, consumer.packing).pairs
    size = (
        dimension if hvp else len(pairs) if consumer.packing == "svec" else dimension**2
    )
    selected = tuple(range(size)) if output_indices is None else tuple(output_indices)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(type(i) is not int or not 0 <= i < size for i in selected)
    ):
        raise ValueError(
            "second derivative output indices must be nonempty, distinct and in bounds"
        )
    if 8 * len(selected) > consumer.memory_budget_bytes:
        raise ValueError("second derivative output tile exceeds its memory budget")
    if len(integral.signature.shells) == 2:
        graph, value, argument, boys_count, differentiate = _one_electron_primal(
            integral, indices, primal_form
        )
    elif integral.operator.family == OperatorFamily.FOUR_CENTER_ERI:
        weights = consumer.weights or WeightDescriptor(
            "raw_unit_weight",
            TensorLayout(
                integral.signature.tensor_indices, integral.signature.component_shape
            ),
        )
        first = replace(
            integral,
            derivative=replace(integral.derivative, order=1),
            contractions=(
                WeightedDerivative(
                    weights,
                    TensorLayout(("center", "xyz"), (len(recovery.centers), 3)),
                    consumer.memory_budget_bytes,
                    output_sign=consumer.output_sign,
                ),
            ),
        )
        kernel = build_weighted_eri_kernel(first, indices, primal_form=primal_form)
        graph, value, argument = kernel.graph, kernel.value, None
        boys_count = integral.maximum_coulomb_order + 1
        differentiate = _eri_differentiate(graph, boys_count - 1)
    else:
        raise ValueError(
            "second derivative lowering supports S/T/V and full four-center Coulomb only"
        )

    independent = tuple(
        (center, axis) for center in recovery.independent for axis in AXES
    )
    gradients = tuple(
        differentiate(value, center, axis) for center, axis in independent
    )

    def row(coordinate):
        center, axis = divmod(coordinate, 3)
        return tuple(
            (3 * i + axis, factor)
            for i, factor in enumerate(recovery.rows[center])
            if factor
        )

    recovered_gradients = tuple(
        graph.sum(factor * gradients[i] for i, factor in row(c))
        for c in range(dimension)
    )
    if hvp:
        # Project v with R.T before differentiation. Its entries remain leaves
        # with zero coordinate response, including when two slots share an atom.
        projected = tuple(
            graph.sum(
                recovery.rows[c][i] * graph.variable(f"direction_{c}_{axis}")
                for c in range(len(recovery.centers))
            )
            for i in range(len(recovery.independent))
            for axis in AXES
        )
        directional = graph.sum(
            v * g for v, g in zip(projected, gradients, strict=True)
        )

        @cache
        def independent_output(i):
            return differentiate(directional, *independent[i])

        outputs = tuple(
            graph.sum(factor * independent_output(i) for i, factor in row(c))
            for c in selected
        )
    else:

        @cache
        def mixed(i, j):
            # Mixed-partial symmetry chooses one generated representative.
            return differentiate(gradients[i], *independent[j])

        outputs = []
        for index in selected:
            i, j = (
                pairs[index] if consumer.packing == "svec" else divmod(index, dimension)
            )
            root = graph.sum(
                a * b * mixed(min(k, coordinate), max(k, coordinate))
                for k, a in row(i)
                for coordinate, b in row(j)
            )
            outputs.append(
                root * (sqrt(2) if consumer.packing == "svec" and i != j else 1)
            )
        outputs = tuple(outputs)
    auxiliary = () if argument is None else (argument,)
    graph, roots = graph.apply_algebra_form(
        (value, *recovered_gradients, *outputs, *auxiliary), output_form
    )
    return SecondDerivativeKernel(
        integral,
        indices,
        graph,
        roots[0],
        roots[1 : 1 + dimension],
        roots[1 + dimension : 1 + dimension + len(outputs)],
        selected,
        recovery,
        roots[-1] if auxiliary else None,
        boys_count,
    )
