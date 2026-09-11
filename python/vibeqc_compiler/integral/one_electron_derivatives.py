"""First S/T/V derivatives and a method-independent external-weight contract.

Differentiate the validated value DAG at generation time. The only special
function rule is dF_n(T) = -F_(n+1)(T) dT; generated device arithmetic contains
no runtime automatic-differentiation objects. Mathematical A/B/C centers stay
distinct until a contraction maps them to physical atoms.
"""

from dataclasses import dataclass, replace

from .blocks import RawBlock, TensorLayout, WeightDescriptor, WeightedDerivative
from .expr import Expr, Graph
from .ir import IntegralIR
from .one_electron_values import (
    build_one_electron_component_kernel,
    build_one_electron_value_ir,
    evaluate_one_electron_primitive,
)
from .shell_spec import AXES


def build_one_electron_derivative_ir(family, angular, *, charge=1.0, weighted=False):
    """Declare raw center/axis/AO blocks or a scalar external-weight gradient.

    Weights have the full unsymmetrized AO layout and are held fixed by this
    derivative. The output is an energy gradient with positive contraction
    sign; the HF adapter supplies -energy-weighted-density for S and density
    for T/V, and negates the final gradient when publishing forces.
    """
    value = build_one_electron_value_ir(family, angular, charge=charge)
    shape = (len(value.operator.centers), 3)
    layout = TensorLayout(("center", "xyz"), shape)
    if weighted:
        contraction = WeightedDerivative(
            WeightDescriptor("external_weight", value.contractions[0].layout),
            layout,
            layout.storage_bytes,
        )
    else:
        layout = TensorLayout(
            layout.indices + value.signature.tensor_indices,
            shape + value.signature.component_shape,
        )
        contraction = RawBlock(layout, layout.storage_bytes)
    return replace(
        value,
        derivative=value.operator.nuclear_derivative(),
        contractions=(contraction,),
    )


@dataclass(frozen=True)
class OneElectronDerivativeKernel:
    """A shared value/derivative DAG in requested mathematical-center order."""

    integral: IntegralIR
    components: tuple[str, str]
    graph: Graph
    value: Expr
    gradients: tuple[tuple[Expr, ...], ...]
    boys_argument: Expr | None
    boys_count: int
    pair_geometry: tuple[tuple[str, Expr], ...]
    hermite_states: tuple[tuple[int, int, int, int], ...]


def build_one_electron_derivative_kernel(integral, components, *, graph=None):
    """Differentiate before cutting the nucleus-independent geometry boundary.

    Graph interning shares Hermite terms, Gaussian decay and prefactors among
    axes/centers/operators. Translation recovers B for S/T and C for V only
    when all required centers are requested. The additional Boys order is at
    most seven for public f/f. Symbolic differentiation avoids explicitly
    raising kinetic Gaussian powers beyond the value recurrence's bound.
    """
    if integral.derivative is None or integral.derivative.order != 1:
        raise ValueError("one-electron lowering requires first nuclear derivatives")
    if any(
        not isinstance(c, (RawBlock, WeightedDerivative)) for c in integral.contractions
    ):
        raise ValueError(
            "one-electron derivatives require raw or external-weight outputs"
        )
    charge = (
        integral.operator.external_centers[0].charge
        if integral.operator.external_centers
        else 1.0
    )
    value_ir = build_one_electron_value_ir(
        integral.operator.family, integral.signature.angular, charge=charge
    )
    # Retain caller declarations so the validated value lowering checks basis
    # representation, center slots and recurrence instead of replacing them.
    value_ir = replace(integral, derivative=None, contractions=value_ir.contractions)
    value = build_one_electron_component_kernel(value_ir, components, graph=graph)
    graph = value.graph
    derivative = integral.derivative
    requested = derivative.requested_centers(integral.operator)
    independent = derivative.independent_centers(integral.operator)
    roots = {}
    for center in independent:
        axes = []
        for axis in AXES:
            variable = graph.variable(f"{'abc'[center]}_{axis}")
            leaves = {}
            if value.boys_argument is not None:
                argument_derivative = graph.differentiate(value.boys_argument, variable)
                leaves = {
                    f"boys_{n}": -graph.variable(f"boys_{n + 1}") * argument_derivative
                    for n in range(value.boys_count)
                }
            axes.append(graph.differentiate(value.value, variable, leaves))
        roots[center] = tuple(axes)
    for center in derivative.recovered_centers(integral.operator):
        roots[center] = tuple(
            -graph.sum(roots[c][axis] for c in independent) for axis in range(3)
        )
    return OneElectronDerivativeKernel(
        integral,
        value.components,
        graph,
        value.value,
        tuple(roots[c] for c in requested),
        value.boys_argument,
        value.boys_count + int(value.boys_argument is not None),
        value.pair_geometry,
        value.hermite_states,
    )


def evaluate_one_electron_derivative_primitive(kernel, exponents, centers):
    """Interpret raw first derivatives for generator diagnostics in atomic units."""
    return tuple(
        tuple(
            evaluate_one_electron_primitive(
                replace(kernel, value=root), exponents, centers
            )
            for root in axis_roots
        )
        for axis_roots in kernel.gradients
    )
