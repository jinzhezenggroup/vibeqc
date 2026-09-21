"""Backend-neutral role-ordered triplet geometry lowered through TensorIR.

Triplet topology is immutable compiler input. Runtime neighbor/topology rebuild
policy is intentionally outside this module; callers must construct a new
contract whenever the admitted triplet set changes.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from fractions import Fraction

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    add,
    constant,
    indexed_gather,
    input_tensor,
    linearize,
    multiply,
    power,
    reduce_sum,
    sqrt,
    transpose_program,
)
from vibeqc_compiler.tensor.ir import rational

from .ir import GeometryIR

SCHEMA = "vibeqc.geometry_triplet"
SCHEMA_VERSION = 1
LOWERING_VERSION = 1
TRIPLET_CONVENTION = "role-ordered-i-j-k-center-j"
TRIPLET_OWNERSHIP = "center-j"


@dataclass(frozen=True)
class TripletTopology:
    """Canonical role-ordered triplets with the second atom as angle center.

    ``(i, j, k)`` is role-ordered, not an unordered set: ``j`` is the center,
    while ``i`` and ``k`` retain caller-defined scientific roles. The tuple list
    itself must be unique and lexicographically ordered so topology identity is
    deterministic without erasing role semantics.
    """

    atom_count: int
    triplets: tuple[tuple[int, int, int], ...]
    convention: str = TRIPLET_CONVENTION
    ownership: str = TRIPLET_OWNERSHIP

    def __post_init__(self) -> None:
        if type(self.atom_count) is not int or self.atom_count < 0:
            raise ValueError("atom_count must be a nonnegative integer")
        triplets = tuple(tuple(triplet) for triplet in self.triplets)
        if any(
            len(triplet) != 3
            or any(type(index) is not int for index in triplet)
            or any(not 0 <= index < self.atom_count for index in triplet)
            or len(set(triplet)) != 3
            for triplet in triplets
        ):
            raise ValueError(
                "triplets must contain three distinct in-range atom indices"
            )
        if tuple(sorted(triplets)) != triplets or len(set(triplets)) != len(triplets):
            raise ValueError(
                "triplet enumeration must be unique and lexicographically ordered"
            )
        if self.convention != TRIPLET_CONVENTION:
            raise ValueError("unsupported triplet convention")
        if self.ownership != TRIPLET_OWNERSHIP:
            raise ValueError("unsupported triplet ownership contract")
        object.__setattr__(self, "triplets", triplets)

    @property
    def owners(self) -> tuple[int, ...]:
        return tuple(j for _, j, _ in self.triplets)

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "atom_count": self.atom_count,
            "triplets": [list(triplet) for triplet in self.triplets],
            "convention": self.convention,
            "ownership": self.ownership,
            "owners": list(self.owners),
        }


@dataclass(frozen=True)
class TripletTensorContext:
    """TensorIR nodes for one fixed role-ordered triplet topology."""

    geometry: GeometryIR
    topology: TripletTopology
    coordinates: Node
    first: Node
    center: Node
    third: Node
    first_vector: Node
    third_vector: Node
    first_squared_distance: Node
    third_squared_distance: Node
    first_distance: Node
    third_distance: Node
    dot: Node
    cosine: Node
    atom_index: Index
    triplet_index: Index
    cartesian_index: Index

    @property
    def triplet_elements(self) -> tuple[tuple[int, int, int], ...]:
        elements = self.geometry.elements
        return tuple(
            (elements[i], elements[j], elements[k])
            for i, j, k in self.topology.triplets
        )


@dataclass(frozen=True)
class TripletProgram:
    """Triplet compiler contract around one shared TensorIR ``Program``."""

    geometry: GeometryIR
    topology: TripletTopology
    program: Program
    parameter_identity: str
    triplet_kind: str
    lowering_version: int = LOWERING_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, GeometryIR):
            raise TypeError("geometry must be GeometryIR")
        if not isinstance(self.topology, TripletTopology):
            raise TypeError("topology must be TripletTopology")
        if not isinstance(self.program, Program):
            raise TypeError("program must be TensorIR Program")
        if self.geometry.atom_count != self.topology.atom_count:
            raise ValueError("geometry/topology atom counts disagree")
        if not isinstance(self.parameter_identity, str) or not self.parameter_identity:
            raise ValueError("parameter_identity must be nonempty")
        if not isinstance(self.triplet_kind, str) or not self.triplet_kind:
            raise ValueError("triplet_kind must be nonempty")
        if self.lowering_version != LOWERING_VERSION:
            raise ValueError("unsupported triplet lowering version")

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "lowering_version": self.lowering_version,
            "equation": self.program.logical_hash,
            "geometry": self.geometry.to_payload(),
            "topology": self.topology.to_payload(),
            "parameter_identity": self.parameter_identity,
            "triplet_kind": self.triplet_kind,
        }

    def validate_execution_identity(self, identity: str) -> None:
        if identity != self.identity:
            raise ValueError("stale triplet compiler execution state")

    def coordinate_jvp(self) -> typing.Any:
        energy = self.program.outputs.get("energy")
        if energy is None or energy.spec.shape != ():
            raise ValueError("coordinate JVP requires scalar output named energy")
        return linearize(
            self.program, [self.geometry.coordinate_name], outputs=["energy"]
        )

    def coordinate_vjp(self) -> typing.Any:
        energy = self.program.outputs.get("energy")
        if energy is None or energy.spec.shape != ():
            raise ValueError("coordinate VJP requires scalar output named energy")
        return transpose_program(
            self.program, ["energy"], inputs=[self.geometry.coordinate_name]
        )


def lower_triplet_geometry(
    geometry: GeometryIR, topology: TripletTopology
) -> TripletTensorContext:
    """Lower role-ordered triplets to gathers and angle geometry in TensorIR."""

    if not isinstance(geometry, GeometryIR) or not isinstance(
        topology, TripletTopology
    ):
        raise TypeError(
            "lower_triplet_geometry requires GeometryIR and TripletTopology"
        )
    if geometry.atom_count != topology.atom_count:
        raise ValueError("geometry/topology atom counts disagree")

    atom_space = IndexSpace("atom", "atom", geometry.atom_count)
    triplet_space = IndexSpace("triplet", "triplet", len(topology.triplets))
    cart_space = IndexSpace("cartesian", "cartesian", 3)
    atom = Index("a", atom_space)
    triplet = Index("t", triplet_space)
    cart = Index("c", cart_space)
    coordinates = input_tensor(
        geometry.coordinate_name,
        TensorSpec(
            (atom, cart), dtype=geometry.dtype, role="input", differentiable=True
        ),
    )

    first_positions = tuple(i for i, _, _ in topology.triplets)
    center_positions = tuple(j for _, j, _ in topology.triplets)
    third_positions = tuple(k for _, _, k in topology.triplets)
    first = indexed_gather(coordinates, 0, first_positions, triplet)
    center = indexed_gather(coordinates, 0, center_positions, triplet)
    third = indexed_gather(coordinates, 0, third_positions, triplet)
    first_vector = add(first, center, coefficients=(1, -1))
    third_vector = add(third, center, coefficients=(1, -1))
    first_squared_distance = reduce_sum(multiply(first_vector, first_vector), (1,))
    third_squared_distance = reduce_sum(multiply(third_vector, third_vector), (1,))
    first_distance = sqrt(first_squared_distance)
    third_distance = sqrt(third_squared_distance)
    dot = reduce_sum(multiply(first_vector, third_vector), (1,))
    denominator = multiply(first_distance, third_distance)
    cosine = multiply(dot, power(denominator, -1))
    return TripletTensorContext(
        geometry,
        topology,
        coordinates,
        first,
        center,
        third,
        first_vector,
        third_vector,
        first_squared_distance,
        third_squared_distance,
        first_distance,
        third_distance,
        dot,
        cosine,
        atom,
        triplet,
        cart,
    )


def triplet_to_system(value: Node, context: TripletTensorContext) -> Node:
    """Reduce one scalar per role-ordered triplet to one system scalar."""

    if value.spec.indices != (context.triplet_index,):
        raise ValueError("triplet_to_system requires one scalar value per triplet")
    return reduce_sum(value, (0,))


def build_triplet_program(
    context: TripletTensorContext,
    triplet_energy: Node,
    *,
    parameter_identity: str,
    triplet_kind: str,
) -> TripletProgram:
    """Wrap a per-triplet scalar equation as a system-energy compiler contract."""

    energy = triplet_to_system(triplet_energy, context)
    provenance = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "lowering_version": LOWERING_VERSION,
        "geometry": context.geometry.to_payload(),
        "topology": context.topology.to_payload(),
        "parameter_identity": parameter_identity,
        "triplet_kind": triplet_kind,
    }
    program = Program({"energy": energy}, provenance=provenance)
    return TripletProgram(
        context.geometry,
        context.topology,
        program,
        parameter_identity,
        triplet_kind,
    )


def cosine_angle_program(
    geometry: GeometryIR,
    topology: TripletTopology,
    coefficients: typing.Any,
    *,
    parameter_identity: str | None = None,
) -> TripletProgram:
    """Qualification potential: ``E = sum_t c_t * cos(i-j-k)``."""

    context = lower_triplet_geometry(geometry, topology)
    if isinstance(coefficients, (int, str, Fraction)):
        coefficients = (coefficients,) * len(topology.triplets)
    else:
        coefficients = tuple(coefficients)
    if len(coefficients) != len(topology.triplets):
        raise ValueError("one coefficient is required per role-ordered triplet")
    exact = tuple(rational(value) for value in coefficients)
    if parameter_identity is None:
        parameter_identity = canonical_hash(
            {"kind": "cosine_angle", "coefficients": exact}
        )
    coefficient = constant(
        coefficients,
        TensorSpec((context.triplet_index,), dtype=geometry.dtype, role="constant"),
    )
    triplet_energy = multiply(coefficient, context.cosine)
    return build_triplet_program(
        context,
        triplet_energy,
        parameter_identity=parameter_identity,
        triplet_kind="cosine-angle",
    )
