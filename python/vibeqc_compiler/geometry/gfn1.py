"""Compiler-owned GFN1-xTB coordination and repulsion equations (#837).

Only pure geometry science lives here. SCC iteration, occupations, eigensolvers,
and convergence policy remain runtime-owned. The GFN1 halogen correction is
intentionally excluded from this pair-only slice because its
donor-neighbor-acceptor angular term requires a triplet/angle topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.tensor import (
    Program,
    TensorSpec,
    add,
    constant,
    divide,
    exp,
    multiply,
    power,
    sqrt,
    transpose_program,
)

from ._gfn1_data import (
    GFN1_COORDINATION_STEEPNESS,
    GFN1_CUTOFF_BOHR,
    GFN1_GEOMETRY_ELEMENT_ROWS,
    GFN1_MINIMUM_DISTANCE_SQUARED_BOHR2,
    GFN1_PARAMETER_JSON_SHA256,
    GFN1_REPULSION_KEXP,
    GFN1_REPULSION_KLIGHT,
)
from .ir import (
    GeometryIR,
    PairCutoff,
    PairTensorContext,
    PairTopology,
    lower_geometry,
    pair_to_atom,
    pair_to_system,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import ArrayLike

    from vibeqc_compiler.tensor.ad_program import VJPProgram
    from vibeqc_compiler.tensor.ir import Node

GFN1_SHORT_RANGE_VERSION = "gfn1-short-range-ir-v1"
GFN1_XTBLOOM_REVISION = "2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3"


@dataclass(frozen=True)
class Gfn1GeometryElement:
    atomic_number: int
    covalent_radius_bohr: float
    arep: float
    zeff: float
    atomic_radius_bohr: float
    xbond: float


GFN1_GEOMETRY_ELEMENTS = MappingProxyType(
    {row[0]: Gfn1GeometryElement(*row) for row in GFN1_GEOMETRY_ELEMENT_ROWS}
)

GFN1_SHORT_RANGE_PARAMETER_IDENTITY = canonical_hash(
    {
        "version": GFN1_SHORT_RANGE_VERSION,
        "xtbloom_revision": GFN1_XTBLOOM_REVISION,
        "gfn1_parameter_json_sha256": GFN1_PARAMETER_JSON_SHA256,
        "coordination": {
            "model": "exp",
            "steepness": float(GFN1_COORDINATION_STEEPNESS).hex(),
            "cutoff_bohr": float(GFN1_CUTOFF_BOHR).hex(),
            "minimum_distance_squared_bohr2": float(
                GFN1_MINIMUM_DISTANCE_SQUARED_BOHR2
            ).hex(),
            "cutoff_inclusive": True,
            "minimum_distance_inclusive": True,
            "maximum_cn_cutoff": None,
        },
        "repulsion": {
            "kexp": float(GFN1_REPULSION_KEXP).hex(),
            "klight": float(GFN1_REPULSION_KLIGHT).hex(),
        },
        "elements": [
            {
                "z": element.atomic_number,
                "covalent_radius_bohr": float(element.covalent_radius_bohr).hex(),
                "arep": float(element.arep).hex(),
                "zeff": float(element.zeff).hex(),
                "atomic_radius_bohr": float(element.atomic_radius_bohr).hex(),
                "xbond": float(element.xbond).hex(),
            }
            for element in GFN1_GEOMETRY_ELEMENTS.values()
        ],
    }
)


def gfn1_element_parameters(atomic_number: int) -> Gfn1GeometryElement:
    if type(atomic_number) is not int:
        raise TypeError("GFN1 atomic number must be an integer")
    try:
        return GFN1_GEOMETRY_ELEMENTS[atomic_number]
    except KeyError as error:
        raise ValueError(
            "GFN1 short-range parameters support atomic numbers 1..86"
        ) from error


def gfn1_geometry(
    elements: Iterable[int], *, coordinate_name: str = "coordinates"
) -> GeometryIR:
    elements = tuple(elements)
    for atomic_number in elements:
        gfn1_element_parameters(atomic_number)
    return GeometryIR(
        elements,
        parameter_identity=GFN1_SHORT_RANGE_PARAMETER_IDENTITY,
        coordinate_name=coordinate_name,
    )


def _validated_coordinates(geometry: GeometryIR, coordinates: ArrayLike) -> np.ndarray:
    if not isinstance(geometry, GeometryIR):
        raise TypeError("geometry must be GeometryIR")
    coordinates = np.asarray(coordinates)
    if coordinates.dtype.kind not in "iuf":
        raise ValueError("GFN1 coordinates must be real numeric arrays")
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (geometry.atom_count, 3):
        raise ValueError("GFN1 coordinates must have shape (atom_count, 3)")
    if not np.isfinite(coordinates).all():
        raise ValueError("GFN1 coordinates must be finite")
    return coordinates


def build_gfn1_pair_topology(
    geometry: GeometryIR,
    coordinates: ArrayLike,
) -> PairTopology:
    """Build the exact nonperiodic GFN1 25-bohr pair set.

    The pinned GFN1 reference skips pairs with r^2 < 1e-12 but evaluates the
    exact threshold. The 25-bohr cutoff is inclusive.
    """

    coordinates = _validated_coordinates(geometry, coordinates)
    cutoff_squared = GFN1_CUTOFF_BOHR * GFN1_CUTOFF_BOHR
    pairs = []
    for first in range(geometry.atom_count):
        for second in range(first + 1, geometry.atom_count):
            displacement = coordinates[second] - coordinates[first]
            distance_squared = float(np.dot(displacement, displacement))
            if distance_squared < GFN1_MINIMUM_DISTANCE_SQUARED_BOHR2:
                continue
            if distance_squared <= cutoff_squared:
                pairs.append((first, second))
    return PairTopology(
        geometry.atom_count,
        tuple(pairs),
        cutoff=PairCutoff(GFN1_CUTOFF_BOHR),
    )


def _require_gfn1_topology(geometry: GeometryIR, topology: PairTopology) -> None:
    if geometry.atom_count != topology.atom_count:
        raise ValueError("GFN1 geometry/topology atom counts disagree")
    if (
        topology.cutoff is None
        or topology.cutoff.radius != GFN1_CUTOFF_BOHR
        or topology.cutoff.switch_start is not None
    ):
        raise ValueError("GFN1 short-range topology requires the sharp 25-bohr cutoff")


def _pair_constant(context: PairTensorContext, values: Iterable[float]) -> Node:
    literals = tuple(repr(float(value)) for value in values)
    if len(literals) != len(context.topology.pairs):
        raise ValueError("GFN1 pair parameter length disagrees with topology")
    return constant(
        literals,
        TensorSpec(
            (context.pair_index,),
            dtype=context.geometry.dtype,
            role="constant",
        ),
    )


def _logistic(argument: Node, one: Node, minus_one: Node) -> Node:
    return divide(one, add(one, exp(multiply(minus_one, argument))))


def _gfn1_pair_terms(context: PairTensorContext) -> tuple[Node, Node]:
    parameters = [
        (
            gfn1_element_parameters(first),
            gfn1_element_parameters(second),
        )
        for first, second in context.pair_elements
    ]
    pair_count = len(context.topology.pairs)
    one = _pair_constant(context, (1.0,) * pair_count)
    minus_one = _pair_constant(context, (-1.0,) * pair_count)
    inverse_distance = power(context.distance, -1)

    radius_sum = _pair_constant(
        context,
        (
            first.covalent_radius_bohr + second.covalent_radius_bohr
            for first, second in parameters
        ),
    )
    ratio = multiply(radius_sum, inverse_distance)
    delta = add(ratio, one, coefficients=(1, -1))
    argument = multiply(
        _pair_constant(
            context,
            (GFN1_COORDINATION_STEEPNESS,) * pair_count,
        ),
        delta,
    )
    pair_coordination = _logistic(argument, one, minus_one)
    coordination = pair_to_atom(pair_coordination, context)

    distance_power = multiply(context.distance, sqrt(context.distance))
    pair_alpha = _pair_constant(
        context,
        ((first.arep * second.arep) ** 0.5 for first, second in parameters),
    )
    pair_charge = _pair_constant(
        context,
        (first.zeff * second.zeff for first, second in parameters),
    )
    decay_argument = multiply(
        minus_one,
        multiply(pair_alpha, distance_power),
    )
    pair_repulsion = multiply(
        multiply(pair_charge, exp(decay_argument)),
        inverse_distance,
    )
    return coordination, pair_repulsion


@dataclass(frozen=True)
class Gfn1ShortRangeProgram:
    """GFN1 CN/repulsion TensorIR graph for one fixed pair topology."""

    geometry: GeometryIR
    topology: PairTopology
    program: Program
    parameter_identity: str = GFN1_SHORT_RANGE_PARAMETER_IDENTITY
    version: str = GFN1_SHORT_RANGE_VERSION

    def __post_init__(self) -> None:
        if self.geometry.parameter_identity != self.parameter_identity:
            raise ValueError("GFN1 geometry parameter identity mismatch")
        _require_gfn1_topology(self.geometry, self.topology)
        if self.version != GFN1_SHORT_RANGE_VERSION:
            raise ValueError("unsupported GFN1 short-range compiler version")

    @property
    def identity(self) -> str:
        return canonical_hash(
            {
                "version": self.version,
                "geometry": self.geometry.to_payload(),
                "topology": self.topology.to_payload(),
                "equation": self.program.logical_hash,
                "parameter_identity": self.parameter_identity,
            }
        )

    def validate_execution_identity(self, identity: str) -> None:
        if identity != self.identity:
            raise ValueError("stale GFN1 geometry compiler execution state")

    def validate_coordinates(self, coordinates: ArrayLike) -> None:
        expected = build_gfn1_pair_topology(self.geometry, coordinates)
        if expected.identity != self.topology.identity:
            raise ValueError("stale GFN1 pair topology for changed coordinates")

    def coordinate_vjp(self, output: str) -> VJPProgram:
        if output not in ("coordination", "repulsion_energy"):
            raise ValueError("unknown GFN1 short-range derivative output")
        return transpose_program(
            self.program,
            [output],
            inputs=[self.geometry.coordinate_name],
        )


def build_gfn1_short_range_program(
    geometry: GeometryIR,
    topology: PairTopology,
) -> Gfn1ShortRangeProgram:
    """Compile GFN1 coordination and repulsion from one TensorIR graph."""

    if not isinstance(geometry, GeometryIR):
        raise TypeError("geometry must be GeometryIR")
    if not isinstance(topology, PairTopology):
        raise TypeError("topology must be PairTopology")
    if geometry.parameter_identity != GFN1_SHORT_RANGE_PARAMETER_IDENTITY:
        raise ValueError(
            "geometry is not bound to the GFN1 short-range parameter identity"
        )
    for atomic_number in geometry.elements:
        gfn1_element_parameters(atomic_number)
    _require_gfn1_topology(geometry, topology)

    context = lower_geometry(geometry, topology)
    coordination, pair_repulsion = _gfn1_pair_terms(context)
    repulsion_energy = pair_to_system(pair_repulsion, context)
    program = Program(
        {
            "coordination": coordination,
            "repulsion_energy": repulsion_energy,
        },
        provenance={
            "kind": "gfn1-short-range",
            "version": GFN1_SHORT_RANGE_VERSION,
            "parameter_identity": GFN1_SHORT_RANGE_PARAMETER_IDENTITY,
            "xtbloom_revision": GFN1_XTBLOOM_REVISION,
            "topology": topology.to_payload(),
            "halogen_lowering": "requires-triplet-ir",
        },
    )
    return Gfn1ShortRangeProgram(geometry, topology, program)
