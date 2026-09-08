"""Raw two-/three-center Coulomb value programs for density fitting.

The tensors are M[P,Q]=(P|Q) and A[mu,nu,P]=(mu nu|P), before any metric
factorization. Every output is one ordinary dense element: neither orbital
pair multiplicities nor inverse-metric factors belong in these kernels.
Primitive Gaussians are unnormalized; contraction and Cartesian normalization
coefficients are supplied by the existing basis layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from itertools import product

from .blocks import RawBlock, TensorLayout
from .expr import Expr, Graph
from .ir import (
    IntegralIR,
    OperatorFamily,
    OperatorSpec,
    TranslationInvariant,
    build_integral_ir,
)
from .shell_class import _component_quantums, _coulomb_derivative, _pair_expansion
from .shell_signature import (
    BasisConvention,
    BasisShell,
    CenterBinding,
    ShellRole,
    ShellSignature,
)
from .shell_spec import AXES, cartesian_components


def build_df_value_ir(
    family: OperatorFamily | str, angular: tuple[int, ...], *, recurrence="subset_wick"
) -> IntegralIR:
    """Declare a Cartesian s/p/d/f raw metric or three-center shell block.

    Auxiliary basis roles and two-/three-center permutation groups are explicit.
    A default auxiliary basis containing g or higher shells is rejected, never
    silently truncated to the public f limit.
    """
    family = OperatorFamily(family)
    if family == OperatorFamily.COULOMB_METRIC:
        roles = (ShellRole.AUXILIARY,) * 2
    elif family == OperatorFamily.THREE_CENTER_ERI:
        roles = (ShellRole.ORBITAL, ShellRole.ORBITAL, ShellRole.AUXILIARY)
    else:
        raise ValueError("DF values require Coulomb metric or three-center ERI")
    angular = tuple(angular)
    if len(angular) != len(roles):
        raise ValueError("DF angular signature does not match its basis roles")
    for l in angular:
        if type(l) is not int or not 0 <= l <= 3:
            raise ValueError(
                "generated DF values support orbital/auxiliary s/p/d/f shells only"
            )
    centers = tuple(range(len(roles)))
    signature = ShellSignature(
        tuple(
            BasisShell(i, i, l, role) for i, (l, role) in enumerate(zip(angular, roles))
        ),
        tuple(CenterBinding(i) for i in centers),
    )
    operator = OperatorSpec(
        family,
        centers,
        invariants=(TranslationInvariant(),),
        permutations=(centers, (1, 0, *centers[2:])),
    )
    layout = TensorLayout(signature.tensor_indices, signature.component_shape)
    return build_integral_ir(
        signature,
        operator=operator,
        contractions=(RawBlock(layout, layout.storage_bytes),),
        recurrence=recurrence,
    )


@dataclass(frozen=True)
class DFComponentKernel:
    """One pruned Hermite/Coulomb DAG, independent of a device schedule.

    ``value`` excludes the radial Coulomb prefactor and Gaussian pair decay.
    Shared geometry inputs have the same meaning for M and A; a metric's first
    Gaussian is a single auxiliary charge, while A has an orbital product.
    """

    integral: IntegralIR
    components: tuple[str, ...]
    graph: Graph
    value: Expr
    coulomb_states: tuple[tuple[int, int, int], ...]


def build_df_component_kernel(
    integral: IntegralIR, components: tuple[str, ...]
) -> DFComponentKernel:
    """Prune raw DF value algebra to the requested Cartesian component.

    A lone auxiliary Gaussian has its product center at its own center. Its
    Hermite expansion therefore has zero center shifts, with a strictly
    positive auxiliary exponent. No fictitious normalized zero-exponent basis
    function enters either the mathematics or the normalization contract.
    """
    signature = integral.signature
    family = integral.operator.family
    if family not in (OperatorFamily.COULOMB_METRIC, OperatorFamily.THREE_CENTER_ERI):
        raise ValueError("not a DF Coulomb value operator")
    if integral.recurrence != "subset_wick":
        raise ValueError("the Hermite component interpreter requires subset_wick IR")
    if integral.derivative is not None or any(
        not isinstance(c, RawBlock) for c in integral.contractions
    ):
        raise ValueError(
            "DF value lowering requires raw values without derivative weights"
        )
    if any(
        s.convention != BasisConvention.CARTESIAN or s.angular > 3
        for s in signature.shells
    ):
        raise ValueError("DF component lowering requires Cartesian s/p/d/f shells")
    components = tuple(components)
    if len(components) != len(signature.shells) or any(
        c not in cartesian_components(l) for c, l in zip(components, signature.angular)
    ):
        raise ValueError("DF component does not match its shell signature")
    graph = Graph()
    zero = {axis: graph.constant(0) for axis in AXES}
    inverse_p = graph.variable("inverse_two_p")
    inverse_q = graph.variable("inverse_two_q")
    if family == OperatorFamily.COULOMB_METRIC:
        first = _component_quantums(components[0], "", zero, zero)
    else:
        shifts = [
            {axis: graph.variable(f"p{slot}_{axis}") for axis in AXES}
            for slot in range(2)
        ]
        first = _component_quantums(components[0], components[1], *shifts)
    second = _component_quantums(components[-1], "", zero, zero)
    left = _pair_expansion(graph, first, inverse_p)
    right = _pair_expansion(graph, second, inverse_q)
    rho = graph.variable("rho")
    difference = {axis: graph.variable(f"difference_{axis}") for axis in AXES}
    boys = tuple(
        graph.variable(f"boys_{n}") for n in range(integral.maximum_coulomb_order + 1)
    )
    states = {}
    terms = []
    for (left_orders, left_coefficient), (right_orders, right_coefficient) in product(
        left, right
    ):
        coefficient = left_coefficient * right_coefficient
        # A zero auxiliary center shift prunes odd/unpaired terms before any
        # Coulomb states or Boys dependencies are constructed.
        if (
            graph.node(coefficient).operation == "constant"
            and graph.node(coefficient).payload == 0
        ):
            continue
        orders = tuple(a + b for a, b in zip(left_orders, right_orders))
        if orders not in states:
            states[orders] = _coulomb_derivative(graph, orders, rho, difference, boys)
        terms.append(
            (-1 if sum(right_orders) % 2 else 1) * coefficient * states[orders]
        )
    return DFComponentKernel(
        integral, components, graph, graph.sum(terms), tuple(sorted(states))
    )


def _reference_boys(argument: float, count: int) -> tuple[float, ...]:
    """Evaluate test/interpreter Boys moments without high-order cancellation."""
    if argument < 24:
        values = []
        for n in range(count):
            term = 1 / (2 * n + 1)
            terms = [term]
            for k in range(256):
                term *= argument / (n + k + 1.5)
                terms.append(term)
                if term <= 2e-17 * math.fsum(terms):
                    break
            values.append(math.exp(-argument) * math.fsum(terms))
        return tuple(values)
    values = [0.5 * math.sqrt(math.pi / argument) * math.erf(math.sqrt(argument))]
    decay = math.exp(-argument)
    for n in range(count - 1):
        values.append(((2 * n + 1) * values[-1] - decay) / (2 * argument))
    return tuple(values)


def evaluate_df_primitive(kernel: DFComponentKernel, exponents, centers) -> float:
    """Interpret one unnormalized primitive integral in Bohr (test/reference use)."""
    exponents = tuple(float(a) for a in exponents)
    centers = tuple(tuple(float(x) for x in r) for r in centers)
    count = len(kernel.components)
    if (
        len(exponents) != count
        or len(centers) != count
        or any(len(r) != 3 for r in centers)
    ):
        raise ValueError("primitive exponents/centers do not match the DF signature")
    if any(not math.isfinite(a) or a <= 0 for a in exponents) or any(
        not math.isfinite(x) for r in centers for x in r
    ):
        raise ValueError(
            "DF primitive exponents must be positive and all inputs finite"
        )
    if count == 2:
        p, q = exponents
        P, Q = centers
        pair_decay = 1.0
    else:
        alpha, beta, q = exponents
        p = alpha + beta
        P = tuple((alpha * a + beta * b) / p for a, b in zip(centers[0], centers[1]))
        Q = centers[2]
        pair_decay = math.exp(
            -alpha
            * beta
            / p
            * math.fsum((a - b) ** 2 for a, b in zip(centers[0], centers[1]))
        )
    rho = p * q / (p + q)
    difference = tuple(a - b for a, b in zip(P, Q))
    variables = {"inverse_two_p": 0.5 / p, "inverse_two_q": 0.5 / q, "rho": rho}
    variables.update({f"difference_{axis}": x for axis, x in zip(AXES, difference)})
    if count == 3:
        for slot in range(2):
            variables.update(
                {f"p{slot}_{axis}": x - a for axis, x, a in zip(AXES, P, centers[slot])}
            )
    variables.update(
        {
            f"boys_{n}": value
            for n, value in enumerate(
                _reference_boys(
                    rho * math.fsum(x * x for x in difference),
                    kernel.integral.maximum_coulomb_order + 1,
                )
            )
        }
    )
    prefactor = 2 * math.pi**2.5 / (p * q * math.sqrt(p + q)) * pair_decay
    return prefactor * kernel.graph.evaluate(kernel.value, variables)


def build_df_axis_moment(a: int, b: int, c: int) -> tuple[Graph, Expr]:
    """Build one-axis Gaussian moments used by a bounded Rys value schedule.

    The first two powers refer to the same electronic coordinate relative to
    the two orbital centers. The last power refers to the auxiliary electronic
    coordinate. For M, b=0. Means and the two-coordinate covariance are external
    root-dependent inputs; recurrence pruning visits only ancestors of (a,b,c).
    """
    if any(type(n) is not int or not 0 <= n <= 3 for n in (a, b, c)):
        raise ValueError("DF axis powers must lie within s/p/d/f")
    graph = Graph()
    means = tuple(graph.variable(f"mean_{i}") for i in range(3))
    xx, xy, yy = (
        graph.variable(name) for name in ("variance_x", "covariance_xy", "variance_y")
    )

    @cache
    def moment(i, j, k):
        if i + j + k == 0:
            return graph.constant(1)
        powers = [i, j, k]
        slot = next(s for s, n in enumerate(powers) if n)
        powers[slot] -= 1
        value = means[slot] * moment(*powers)
        for partner, n in enumerate(powers):
            if n:
                covariance = (
                    yy
                    if slot == partner == 2
                    else xy
                    if (slot == 2) != (partner == 2)
                    else xx
                )
                lower = list(powers)
                lower[partner] -= 1
                value += n * covariance * moment(*lower)
        return value

    return graph, moment(a, b, c)
