"""Bounded arbitrary-weight ERI values and first derivatives in the shared DAG.

Weights are fixed cotangents for this partial derivative. Their own coordinate
response belongs to the caller's response/Lagrangian equations. Shell-center
slots remain independent through differentiation and translation recovery;
physical atom accumulation is a later consumer operation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product

from .blocks import TensorLayout, WeightDescriptor, WeightedDerivative
from .expr import AlgebraForm, Expr, Graph
from .ir import (
    FOUR_CENTER_ERI_OPERATOR,
    IntegralIR,
    OperatorFamily,
    OperatorSpec,
    build_integral_ir,
)
from .shell_class import _component_quantums, _coulomb_derivative, _pair_expansion
from .shell_signature import BasisConvention, ShellSignature
from .shell_spec import AXES, ShellClassSpec

CENTERS = ("first", "second", "third", "fourth")


def build_weighted_eri_ir(
    angular: tuple[int, int, int, int],
    *,
    memory_budget_bytes=4 * 1024**2,
    operator: OperatorSpec = FOUR_CENTER_ERI_OPERATOR,
) -> IntegralIR:
    """Describe a full Cartesian shell tile with arbitrary ordered weights.

    A unique-orbit caller must fold its orbit weights explicitly before
    supplying them. This request itself neither restricts AO indices to a
    triangular domain nor applies multiplicities or an HF density formula.
    """
    angular = tuple(angular)
    if len(angular) != 4 or any(type(l) is not int or not 0 <= l <= 3 for l in angular):
        raise ValueError("weighted ERI lowering supports four s/p/d/f shells")
    spec = ShellClassSpec("".join("spdf"[l] for l in angular), angular)
    signature = ShellSignature.from_shell_class(spec)
    consumer = WeightedDerivative(
        WeightDescriptor(
            "external_eri_weights",
            TensorLayout(signature.tensor_indices, signature.component_shape),
        ),
        TensorLayout(("center", "xyz"), (4, 3)),
        memory_budget_bytes,
    )
    return build_integral_ir(
        spec,
        operator=operator,
        derivative=operator.nuclear_derivative(),
        contractions=(consumer,),
    )


def canonical_range_weighted_eri_ir(integral):
    """Validate the public request, retaining unit factors in the native consumer."""
    if not integral.operator.range_separated or integral.recurrence != "subset_wick":
        raise ValueError(
            "generated weighted execution requires a range subset_wick operator"
        )
    if integral.derivative is None or integral.derivative.order != 1:
        raise ValueError(
            "generated weighted execution requires first nuclear derivatives"
        )
    if len(integral.contractions) != 1 or not isinstance(
        integral.contractions[0], WeightedDerivative
    ):
        raise ValueError(
            "generated weighted execution requires the external-weight consumer"
        )
    if integral.operator.centers != (0, 1, 2, 3) or tuple(
        s.center for s in integral.signature.shells
    ) != (0, 1, 2, 3):
        raise ValueError(
            "generated weighted execution requires four canonical shell slots"
        )
    return build_weighted_eri_ir(integral.signature.angular, operator=integral.operator)


@dataclass(frozen=True)
class WeightedEriKernel:
    """One explicit component subset with no raw four-index derivative buffer."""

    integral: IntegralIR
    spec: ShellClassSpec
    component_indices: tuple[int, ...]
    graph: Graph
    value: Expr
    gradients: tuple[tuple[Expr, Expr, Expr], ...]
    coulomb_states: tuple[tuple[int, int, int], ...]


def build_weighted_eri_kernel(
    integral: IntegralIR,
    component_indices: tuple[int, ...] | None = None,
    *,
    primal_form: AlgebraForm = AlgebraForm.FACTORED_NARY,
) -> WeightedEriKernel:
    """Precontract Hermite coefficients, then differentiate the weighted scalar.

    Common Coulomb states and pair expansions are built once per bounded
    subset. Differentiating after weight contraction preserves sharing across
    Cartesian outputs, including psss's weighted PA/PQ dot products. A caller
    can partition a large class into explicit subsets of at most 64 components;
    this code-size bound is a lowering choice, not a mathematical domain limit.
    Uncompiled subsets retain the unscreened native external-weight fallback.
    ``primal_form`` permits algebra/AD commutation checks; its default preserves
    the established first-derivative graph and emitted source.
    """
    if (
        integral.operator.family
        not in (
            OperatorFamily.FOUR_CENTER_ERI,
            OperatorFamily.LONG_RANGE_ERI,
            OperatorFamily.SHORT_RANGE_ERI,
        )
        or integral.operator.centers != (0, 1, 2, 3)
        or tuple(s.center for s in integral.signature.shells) != (0, 1, 2, 3)
    ):
        raise ValueError("weighted ERI lowering requires four ERI center slots")
    if integral.derivative is None or integral.derivative.order != 1:
        raise ValueError("weighted ERI lowering requires first derivatives")
    if integral.recurrence != "subset_wick":
        raise ValueError("weighted Hermite lowering requires subset_wick recurrence")
    if len(integral.contractions) != 1 or not isinstance(
        integral.contractions[0], WeightedDerivative
    ):
        raise ValueError("weighted ERI lowering requires an external-weight consumer")
    signature = integral.signature
    if any(
        s.convention != BasisConvention.CARTESIAN or s.angular > 3
        for s in signature.shells
    ):
        raise ValueError(
            "pull public weights back to Cartesian s/p/d/f components first"
        )
    spec = ShellClassSpec(
        "".join("spdf"[l] for l in signature.angular), signature.angular
    )
    consumer = integral.contractions[0]
    if consumer.weights.layout.shape != signature.component_shape:
        raise ValueError(
            "kernel input describes a full shell weight layout; select a bounded component subset explicitly"
        )
    indices = (
        tuple(range(spec.component_count))
        if component_indices is None
        else tuple(component_indices)
    )
    if not indices or len(indices) > 64:
        raise ValueError(
            "select between one and 64 explicit weighted components per lowering"
        )
    if len(set(indices)) != len(indices) or any(
        type(i) is not int or not 0 <= i < spec.component_count for i in indices
    ):
        raise ValueError(
            "weighted component indices must be unique and within the shell"
        )

    graph = Graph()
    inverse_p, inverse_q, rho = (
        graph.variable(n) for n in ("inverse_two_p", "inverse_two_q", "rho")
    )
    shifts = {
        prefix: {axis: graph.variable(f"{prefix}_{axis}") for axis in AXES}
        for prefix in ("pa", "pb", "qc", "qd")
    }
    difference = {axis: graph.variable(f"difference_{axis}") for axis in AXES}
    boys = tuple(
        graph.variable(f"boys_{n}") for n in range(integral.maximum_coulomb_order + 1)
    )
    coefficients = defaultdict(list)
    first_cache, second_cache = {}, {}
    for index in indices:
        a, b, c, d = spec.components[index]
        if (a, b) not in first_cache:
            first_cache[a, b] = _pair_expansion(
                graph, _component_quantums(a, b, shifts["pa"], shifts["pb"]), inverse_p
            )
        if (c, d) not in second_cache:
            second_cache[c, d] = _pair_expansion(
                graph, _component_quantums(c, d, shifts["qc"], shifts["qd"]), inverse_q
            )
        weight = graph.variable(f"component_weight_{index}")
        for (left, left_coefficient), (right, right_coefficient) in product(
            first_cache[a, b], second_cache[c, d]
        ):
            orders = tuple(i + j for i, j in zip(left, right))
            coefficients[orders].append(
                weight
                * left_coefficient
                * right_coefficient
                * (-1 if sum(right) % 2 else 1)
            )
    value = graph.variable("prefactor") * graph.sum(
        graph.sum(terms) * _coulomb_derivative(graph, orders, rho, difference, boys)
        for orders, terms in sorted(coefficients.items())
    )
    scale = consumer.output_sign * consumer.weights.sign * consumer.weights.prefactor
    # Factor the scalar before AD so component weights share expensive states.
    graph, (value,) = graph.apply_algebra_form(
        (scale * value,), AlgebraForm(primal_form)
    )
    prefactor = graph.variable("prefactor")
    rho = graph.variable("rho")
    boys = tuple(
        graph.variable(f"boys_{n}") for n in range(integral.maximum_coulomb_order + 1)
    )
    gradients = {}
    for center in integral.independent_derivative_centers:
        pair_scale = graph.variable(f"{CENTERS[center]}_product_scale")
        difference_scale = pair_scale if center < 2 else -pair_scale
        rows = []
        for axis in AXES:
            leaves = {
                "prefactor": prefactor
                * graph.variable(f"decay_{CENTERS[center]}_{axis}"),
                f"difference_{axis}": difference_scale,
            }
            prefixes = ("pa", "pb") if center < 2 else ("qc", "qd")
            for slot, prefix in enumerate(prefixes):
                leaves[f"{prefix}_{axis}"] = pair_scale - int(slot == center % 2)
            argument_derivative = (
                2 * rho * graph.variable(f"difference_{axis}") * difference_scale
            )
            leaves.update(
                {
                    f"boys_{n}": -boys[n + 1] * argument_derivative
                    for n in range(integral.maximum_coulomb_order)
                }
            )
            # This seed is absent from the value; explicit leaf responses carry
            # the geometry chain rule and keep every external weight constant.
            rows.append(
                graph.differentiate(
                    value, graph.variable(f"coordinate_{center}_{axis}"), leaves
                )
            )
        gradients[center] = tuple(rows)
    for center in integral.recovered_derivative_centers:
        gradients[center] = tuple(
            -graph.sum(
                gradients[i][axis] for i in integral.independent_derivative_centers
            )
            for axis in range(3)
        )
    requested = set(integral.requested_derivative_centers)
    result = tuple(
        gradients[c] if c in requested else tuple(graph.constant(0) for _ in AXES)
        for c in range(4)
    )
    return WeightedEriKernel(
        integral, spec, indices, graph, value, result, tuple(sorted(coefficients))
    )
