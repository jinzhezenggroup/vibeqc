"""Compiler-owned GFN1-xTB short-range geometry equations (#837, #853).

Only pure geometry science lives here. SCC iteration, occupations, eigensolvers,
and convergence policy remain runtime-owned. The halogen correction uses the
generic role-ordered TripletIR graph; no handwritten derivative lives here.
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
    GFN1_HALOGEN_DAMPING,
    GFN1_HALOGEN_RADIUS_SCALE,
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
from .triplet import (
    TripletProgram,
    TripletTensorContext,
    TripletTopology,
    build_triplet_program,
    lower_triplet_geometry,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import ArrayLike

    from vibeqc_compiler.tensor.ad_program import VJPProgram
    from vibeqc_compiler.tensor.ir import Node

GFN1_SHORT_RANGE_VERSION = "gfn1-short-range-ir-v1"
GFN1_XTBLOOM_REVISION = "2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3"
GFN1_HALOGEN_VERSION = "gfn1-halogen-triplet-ir-v1"
GFN1_HALOGEN_CUTOFF_BOHR = 20.0
GFN1_HALOGEN_DONOR_ELEMENTS = (17, 35, 53, 85)
GFN1_HALOGEN_ACCEPTOR_ELEMENTS = (7, 8, 15, 16)
GFN1_TBLITE_REVISION = "fa8a4416e8fe093d0075bc10ac875494c2a449a9"
GFN1_TBLITE_HALOGEN_SHA256 = (
    "ed3469a1e07d95d75bb09b1a4615616a9416aff425c9cc46ae057dca450ad449"
)


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

GFN1_HALOGEN_PARAMETER_IDENTITY = canonical_hash(
    {
        "version": GFN1_HALOGEN_VERSION,
        "xtbloom_revision": GFN1_XTBLOOM_REVISION,
        "gfn1_parameter_json_sha256": GFN1_PARAMETER_JSON_SHA256,
        "tblite_revision": GFN1_TBLITE_REVISION,
        "tblite_halogen_source": {
            "path": "src/tblite/classical/halogen.f90",
            "sha256": GFN1_TBLITE_HALOGEN_SHA256,
        },
        "roles": {
            "triplet": ("nearest-neighbor", "halogen-donor", "acceptor"),
            "center": "halogen-donor",
            "donor_elements": GFN1_HALOGEN_DONOR_ELEMENTS,
            "acceptor_elements": GFN1_HALOGEN_ACCEPTOR_ELEMENTS,
            "nearest_neighbor_tie_break": "lowest-atom-index",
            "degenerate_neighbor_acceptor": "omitted-identically-zero",
        },
        "cutoff_bohr": float(GFN1_HALOGEN_CUTOFF_BOHR).hex(),
        "cutoff_inclusive": True,
        "damping": float(GFN1_HALOGEN_DAMPING).hex(),
        "radius_scale": float(GFN1_HALOGEN_RADIUS_SCALE).hex(),
        "angular_exponent": 6,
        "radial_exponents": (6, 12),
        "elements": [
            {
                "z": element.atomic_number,
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
            "separate_halogen_lowering_version": GFN1_HALOGEN_VERSION,
        },
    )
    return Gfn1ShortRangeProgram(geometry, topology, program)


def build_gfn1_halogen_topology(
    geometry: GeometryIR,
    coordinates: ArrayLike,
) -> TripletTopology:
    """Build pinned nonperiodic ``(neighbor, donor, acceptor)`` triplets.

    tblite enumerates every halogen-donor/acceptor pair within an inclusive
    20-bohr cutoff, then chooses the donor's closest positive-distance atom.
    The strict nearest-neighbor comparison makes the lowest atom index the tie
    break. If that neighbor is the acceptor, the angular factor is identically
    zero, so the repeated-role term is omitted rather than violating the
    three-distinct-atom TripletIR contract.
    """

    if not isinstance(geometry, GeometryIR):
        raise TypeError("geometry must be GeometryIR")
    if geometry.parameter_identity != GFN1_SHORT_RANGE_PARAMETER_IDENTITY:
        raise ValueError(
            "geometry is not bound to the canonical GFN1 parameter identity"
        )
    coordinates = _validated_coordinates(geometry, coordinates)
    triplets: list[tuple[int, int, int]] = []
    for donor, donor_element in enumerate(geometry.elements):
        if donor_element not in GFN1_HALOGEN_DONOR_ELEMENTS:
            continue
        acceptors = []
        for acceptor, acceptor_element in enumerate(geometry.elements):
            if acceptor_element not in GFN1_HALOGEN_ACCEPTOR_ELEMENTS:
                continue
            distance = float(np.linalg.norm(coordinates[acceptor] - coordinates[donor]))
            if distance == 0.0:
                raise ValueError("GFN1 halogen donor/acceptor coincidence is undefined")
            if distance <= GFN1_HALOGEN_CUTOFF_BOHR:
                acceptors.append(acceptor)
        if not acceptors:
            continue

        nearest = None
        nearest_distance = np.inf
        for candidate in range(geometry.atom_count):
            distance = float(
                np.linalg.norm(coordinates[candidate] - coordinates[donor])
            )
            if distance > 0.0 and distance < nearest_distance:
                nearest = candidate
                nearest_distance = distance
        if nearest is None:
            raise ValueError("GFN1 halogen donor has no positive-distance neighbor")

        triplets.extend(
            (nearest, donor, acceptor) for acceptor in acceptors if acceptor != nearest
        )
    return TripletTopology(geometry.atom_count, tuple(sorted(triplets)))


def _require_gfn1_halogen_topology(
    geometry: GeometryIR, topology: TripletTopology
) -> None:
    if not isinstance(topology, TripletTopology):
        raise TypeError("topology must be TripletTopology")
    if geometry.atom_count != topology.atom_count:
        raise ValueError("GFN1 geometry/topology atom counts disagree")
    donor_acceptors: set[tuple[int, int]] = set()
    for _neighbor, donor, acceptor in topology.triplets:
        if geometry.elements[donor] not in GFN1_HALOGEN_DONOR_ELEMENTS:
            raise ValueError("GFN1 halogen topology center must be a halogen donor")
        if geometry.elements[acceptor] not in GFN1_HALOGEN_ACCEPTOR_ELEMENTS:
            raise ValueError("GFN1 halogen topology third role must be an acceptor")
        pair = (donor, acceptor)
        if pair in donor_acceptors:
            raise ValueError("GFN1 halogen topology repeats a donor/acceptor pair")
        donor_acceptors.add(pair)


def _triplet_constant(context: TripletTensorContext, values: Iterable[float]) -> Node:
    literals = tuple(repr(float(value)) for value in values)
    if len(literals) != len(context.topology.triplets):
        raise ValueError("GFN1 halogen parameter length disagrees with topology")
    return constant(
        literals,
        TensorSpec(
            (context.triplet_index,),
            dtype=context.geometry.dtype,
            role="constant",
        ),
    )


def _gfn1_halogen_triplet_energy(context: TripletTensorContext) -> Node:
    parameters = [
        (
            gfn1_element_parameters(donor),
            gfn1_element_parameters(acceptor),
        )
        for _neighbor, donor, acceptor in context.triplet_elements
    ]
    count = len(parameters)
    one = _triplet_constant(context, (1.0,) * count)
    half = _triplet_constant(context, (0.5,) * count)
    damping = _triplet_constant(context, (GFN1_HALOGEN_DAMPING,) * count)
    strength = _triplet_constant(
        context, (donor.xbond for donor, _acceptor in parameters)
    )
    radius = _triplet_constant(
        context,
        (
            GFN1_HALOGEN_RADIUS_SCALE
            * (donor.atomic_radius_bohr + acceptor.atomic_radius_bohr)
            for donor, acceptor in parameters
        ),
    )

    angular_base = add(half, multiply(half, context.cosine), coefficients=(1, -1))
    angular_squared = multiply(angular_base, angular_base)
    angular = multiply(angular_squared, multiply(angular_squared, angular_squared))
    radius_ratio = divide(radius, context.third_distance)
    ratio_six = power(radius_ratio, 6)
    ratio_twelve = multiply(ratio_six, ratio_six)
    radial = divide(
        add(ratio_twelve, multiply(damping, ratio_six), coefficients=(1, -1)),
        add(one, ratio_twelve),
    )
    return multiply(multiply(angular, strength), radial)


@dataclass(frozen=True)
class Gfn1HalogenGeometryProgram:
    """GFN1 halogen energy and generated Cartesian derivatives for one topology."""

    geometry: GeometryIR
    topology: TripletTopology
    triplet_program: TripletProgram
    parameter_identity: str = GFN1_HALOGEN_PARAMETER_IDENTITY
    version: str = GFN1_HALOGEN_VERSION

    def __post_init__(self) -> None:
        if self.parameter_identity != GFN1_HALOGEN_PARAMETER_IDENTITY:
            raise ValueError("canonical GFN1 halogen parameter identity required")
        if self.geometry.parameter_identity != GFN1_SHORT_RANGE_PARAMETER_IDENTITY:
            raise ValueError("GFN1 halogen geometry parameter identity mismatch")
        _require_gfn1_halogen_topology(self.geometry, self.topology)
        if self.triplet_program.geometry != self.geometry:
            raise ValueError("GFN1 halogen TripletIR geometry mismatch")
        if self.triplet_program.topology != self.topology:
            raise ValueError("GFN1 halogen TripletIR topology mismatch")
        if self.triplet_program.parameter_identity != self.parameter_identity:
            raise ValueError("GFN1 halogen TripletIR parameter identity mismatch")
        if self.triplet_program.triplet_kind != "gfn1-halogen-neighbor-donor-acceptor":
            raise ValueError("GFN1 halogen TripletIR kind mismatch")
        if self.version != GFN1_HALOGEN_VERSION:
            raise ValueError("unsupported GFN1 halogen compiler version")

    @property
    def program(self) -> Program:
        return self.triplet_program.program

    @property
    def identity(self) -> str:
        return canonical_hash(
            {
                "version": self.version,
                "triplet_program": self.triplet_program.to_payload(),
                "parameter_identity": self.parameter_identity,
            }
        )

    def validate_execution_identity(self, identity: str) -> None:
        if identity != self.identity:
            raise ValueError("stale GFN1 halogen compiler execution state")

    def validate_coordinates(self, coordinates: ArrayLike) -> None:
        expected = build_gfn1_halogen_topology(self.geometry, coordinates)
        if expected.identity != self.topology.identity:
            raise ValueError("stale GFN1 halogen topology for changed coordinates")

    def coordinate_jvp(self) -> object:
        return self.triplet_program.coordinate_jvp()

    def coordinate_vjp(self) -> object:
        return self.triplet_program.coordinate_vjp()


def build_gfn1_halogen_geometry_program(
    geometry: GeometryIR,
    topology: TripletTopology,
) -> Gfn1HalogenGeometryProgram:
    """Lower the pinned GFN1 halogen scalar expression through TripletIR."""

    if not isinstance(geometry, GeometryIR):
        raise TypeError("geometry must be GeometryIR")
    if geometry.parameter_identity != GFN1_SHORT_RANGE_PARAMETER_IDENTITY:
        raise ValueError(
            "geometry is not bound to the canonical GFN1 parameter identity"
        )
    _require_gfn1_halogen_topology(geometry, topology)
    context = lower_triplet_geometry(geometry, topology)
    triplet_program = build_triplet_program(
        context,
        _gfn1_halogen_triplet_energy(context),
        parameter_identity=GFN1_HALOGEN_PARAMETER_IDENTITY,
        triplet_kind="gfn1-halogen-neighbor-donor-acceptor",
    )
    return Gfn1HalogenGeometryProgram(geometry, topology, triplet_program)
