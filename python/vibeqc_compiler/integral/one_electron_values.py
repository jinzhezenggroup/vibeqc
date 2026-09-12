"""Scalar one-electron value DAGs shared by S/T/V shell-pair lowerings.

The graph represents unnormalized primitive Cartesian Gaussians. Contracted
basis coefficients and component normalization belong to the existing basis
layer. S and T use a common pruned Hermite recurrence; V additionally uses the
existing Coulomb derivative algebra and explicit Boys inputs at its declared
external nuclear center. No device schedule or runtime AD objects enter here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from itertools import product

from .blocks import RawBlock, TensorLayout
from .df_values import _reference_boys
from .expr import Expr, Graph
from .ir import (
    IntegralIR,
    NuclearCenter,
    OperatorFamily,
    OperatorSpec,
    TranslationInvariant,
    build_integral_ir,
)
from .shell_class import _coulomb_derivative
from .shell_signature import BasisConvention, BasisShell, CenterBinding, ShellSignature
from .shell_spec import AXES, cartesian_components

_FAMILIES = (
    OperatorFamily.OVERLAP,
    OperatorFamily.KINETIC,
    OperatorFamily.NUCLEAR_ATTRACTION,
)


def build_one_electron_value_ir(family, angular, *, charge=1.0):
    """Declare one Cartesian s/p/d/f/g pair and its raw two-index value block.

    Attraction has a third, independent mathematical center with positive
    nuclear charge. Its physical minus sign is part of V. Basis-center
    coincidence or shared atom ownership never removes this center from IR.
    """
    family = OperatorFamily(family)
    angular = tuple(angular)
    if family not in _FAMILIES:
        raise ValueError("one-electron values require overlap, kinetic or attraction")
    if len(angular) != 2 or any(type(l) is not int or not 0 <= l <= 4 for l in angular):
        raise ValueError("one-electron values require two public s/p/d/f/g shells")
    attraction = family == OperatorFamily.NUCLEAR_ATTRACTION
    centers = (0, 1, 2) if attraction else (0, 1)
    signature = ShellSignature(
        tuple(BasisShell(i, i, l) for i, l in enumerate(angular)),
        tuple(CenterBinding(i) for i in centers),
    )
    operator = OperatorSpec(
        family,
        centers,
        invariants=(TranslationInvariant(),),
        external_centers=(NuclearCenter(2, charge),) if attraction else (),
        permutations=((0, 1), (1, 0)),
    )
    layout = TensorLayout(signature.tensor_indices, signature.component_shape)
    return build_integral_ir(
        signature,
        operator=operator,
        contractions=(RawBlock(layout, layout.storage_bytes),),
        recurrence="hermite",
    )


@dataclass(frozen=True)
class OneElectronComponentKernel:
    """A scalar root plus the scientific IR that fixes its exact conventions.

    Only V needs external Boys moments. ``boys_argument`` is itself a graph
    root, so a backend can evaluate the shared geometry once, fill the required
    moments, then evaluate ``value``. The scalar graph includes Gaussian decay,
    the radial prefactor and (for attraction) the physical nuclear-charge sign.
    """

    integral: IntegralIR
    components: tuple[str, str]
    graph: Graph
    value: Expr
    boys_argument: Expr | None
    boys_count: int
    hermite_states: tuple[tuple[int, int, int, int], ...]
    pair_geometry: tuple[tuple[str, Expr], ...]


def build_one_electron_component_kernel(integral, components, *, graph=None):
    """Generate only recurrence ancestors needed by one S/T/V component.

    Kinetic raising may need an internal ket power of six for a public g
    shell. This does not widen the public basis contract or output dimensions.
    Recurrence memoization and Graph interning share T's overlap subexpressions;
    the scalar emitter later prunes constants and nodes unreachable from roots.
    An optional shared graph lets multi-operator consumers intern S/T together.
    ``pair_geometry`` identifies nucleus-independent roots that a lowering can
    hoist outside the nuclear traversal without changing the scientific graph.
    """
    family = integral.operator.family
    if family not in _FAMILIES or integral.recurrence != "hermite":
        raise ValueError("one-electron Hermite value IR required")
    if integral.derivative is not None or any(
        not isinstance(c, RawBlock) for c in integral.contractions
    ):
        raise ValueError("one-electron component lowering requires raw value outputs")
    if len(integral.signature.shells) != 2 or any(
        s.angular > 4 or s.convention != BasisConvention.CARTESIAN
        for s in integral.signature.shells
    ):
        raise ValueError("one-electron lowering requires Cartesian s/p/d/f/g shells")
    expected_centers = (
        (0, 1, 2) if family == OperatorFamily.NUCLEAR_ATTRACTION else (0, 1)
    )
    if integral.operator.centers != expected_centers or tuple(
        shell.center for shell in integral.signature.shells
    ) != (0, 1):
        raise ValueError(
            "one-electron lowering requires basis slots A/B and independent nuclear C"
        )
    if (
        family == OperatorFamily.NUCLEAR_ATTRACTION
        and integral.operator.external_centers[0].center != 2
    ):
        raise ValueError("the external nuclear charge must belong to center C")
    components = tuple(components)
    if len(components) != 2 or any(
        c not in cartesian_components(l)
        for c, l in zip(components, integral.signature.angular)
    ):
        raise ValueError("Cartesian components do not match the shell-pair signature")
    graph = Graph() if graph is None else graph
    alpha, beta = graph.variable("alpha"), graph.variable("beta")
    p = alpha + beta
    inverse_p = 1 / p
    a = tuple(graph.variable(f"a_{axis}") for axis in AXES)
    b = tuple(graph.variable(f"b_{axis}") for axis in AXES)
    ab = tuple(x - y for x, y in zip(a, b))
    # Relative shifts avoid forming a large absolute Gaussian product center.
    pa = tuple(-beta * inverse_p * x for x in ab)
    pb = tuple(alpha * inverse_p * x for x in ab)
    decay = graph.exponential(-alpha * beta * inverse_p * graph.sum(x * x for x in ab))
    zero = graph.constant(0)
    powers = tuple(tuple(c.count(axis) for axis in AXES) for c in components)
    states = set()

    @cache
    def hermite(axis, i, j, t):
        if i < 0 or j < 0 or t < 0 or t > i + j:
            return zero
        states.add((axis, i, j, t))
        if i == j == 0:
            return graph.constant(1)
        if i:
            return (
                0.5 * inverse_p * hermite(axis, i - 1, j, t - 1)
                + pa[axis] * hermite(axis, i - 1, j, t)
                + (t + 1) * hermite(axis, i - 1, j, t + 1)
            )
        return (
            0.5 * inverse_p * hermite(axis, i, j - 1, t - 1)
            + pb[axis] * hermite(axis, i, j - 1, t)
            + (t + 1) * hermite(axis, i, j - 1, t + 1)
        )

    prefactor = decay * graph.power(math.pi * inverse_p, 1.5)

    def overlap(first, second):
        return prefactor * graph.multiply_many(
            hermite(k, i, j, 0) for k, (i, j) in enumerate(zip(first, second))
        )

    boys_argument = None
    boys_count = 0
    first, second = powers
    if family == OperatorFamily.OVERLAP:
        value = overlap(first, second)
    elif family == OperatorFamily.KINETIC:
        value = beta * (2 * sum(second) + 3) * overlap(first, second)
        for axis in range(3):
            raised = list(second)
            raised[axis] += 2
            value -= 2 * beta * beta * overlap(first, tuple(raised))
            if second[axis] >= 2:
                lowered = list(second)
                lowered[axis] -= 2
                value -= (
                    0.5
                    * second[axis]
                    * (second[axis] - 1)
                    * overlap(first, tuple(lowered))
                )
    else:
        c = tuple(graph.variable(f"c_{axis}") for axis in AXES)
        # Subtract the two absolute centers before adding the relative product
        # shift, retaining precision when the molecule has a large common origin.
        difference = {axis: a[k] - c[k] + pa[k] for k, axis in enumerate(AXES)}
        boys_argument = p * graph.sum(x * x for x in difference.values())
        boys_count = sum(first) + sum(second) + 1
        boys = tuple(graph.variable(f"boys_{n}") for n in range(boys_count))
        terms = []
        for orders in product(*(range(i + j + 1) for i, j in zip(first, second))):
            coefficient = graph.multiply_many(
                hermite(k, first[k], second[k], t) for k, t in enumerate(orders)
            )
            terms.append(
                coefficient * _coulomb_derivative(graph, orders, p, difference, boys)
            )
        charge = integral.operator.external_centers[0].charge
        value = -charge * 2 * math.pi * inverse_p * decay * graph.sum(terms)
    return OneElectronComponentKernel(
        integral,
        components,
        graph,
        value,
        boys_argument,
        boys_count,
        tuple(sorted(states)),
        tuple(
            [
                ("p", p),
                ("inverse_p", inverse_p),
                ("decay", decay),
                ("overlap_scale", prefactor),
            ]
            + [(f"pa_{axis}", pa[k]) for k, axis in enumerate(AXES)]
            + [(f"pb_{axis}", pb[k]) for k, axis in enumerate(AXES)]
        ),
    )


def evaluate_one_electron_primitive(kernel, exponents, centers):
    """CPU interpreter for one unnormalized primitive value in atomic units.

    Attraction positions include independent A/B/C; S/T positions include A/B.
    This is a generator diagnostic, not a production HF integral provider.
    """
    exponents = tuple(float(x) for x in exponents)
    centers = tuple(tuple(float(x) for x in position) for position in centers)
    expected = 3 if kernel.boys_argument is not None else 2
    if (
        len(exponents) != 2
        or len(centers) != expected
        or any(len(x) != 3 for x in centers)
    ):
        raise ValueError("one-electron primitive dimensions do not match the operator")
    if any(not math.isfinite(x) or x <= 0 for x in exponents) or any(
        not math.isfinite(x) for r in centers for x in r
    ):
        raise ValueError("primitive exponents must be positive and all inputs finite")
    if not math.isfinite(sum(exponents)):
        raise ValueError("primitive exponent sum overflows")
    variables = dict(zip(("alpha", "beta"), exponents))
    for name, position in zip("abc", centers):
        variables.update({f"{name}_{axis}": x for axis, x in zip(AXES, position)})
    if kernel.boys_argument is not None:
        argument = kernel.graph.evaluate(kernel.boys_argument, variables)
        variables.update(
            {
                f"boys_{n}": x
                for n, x in enumerate(_reference_boys(argument, kernel.boys_count))
            }
        )
    return kernel.graph.evaluate(kernel.value, variables)
