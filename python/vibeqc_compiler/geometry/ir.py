"""Backend-neutral geometry/pair IR lowering through shared TensorIR.

Pair topology is immutable compiler input. Runtime neighbor-list rebuild policy
is intentionally outside this module; callers must construct a new topology
contract when the pair set or cutoff contract changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    add,
    constant,
    einsum,
    gather,
    input_tensor,
    linearize,
    multiply,
    power,
    reduce_sum,
    reshape,
    sqrt,
    transpose_program,
)
from vibeqc_compiler.tensor.ir import rational

SCHEMA = "vibeqc.geometry_pair"
SCHEMA_VERSION = 1
LOWERING_VERSION = 1
PAIR_OWNERSHIP = "canonical-undirected-lower-atom"


@dataclass(frozen=True)
class PairCutoff:
    """Static cutoff/switch contract; rebuild policy remains a runtime concern."""

    radius: float
    switch_start: float | None = None

    def __post_init__(self) -> None:
        radius = float(self.radius)
        switch = None if self.switch_start is None else float(self.switch_start)
        if not isfinite(radius) or radius <= 0.0:
            raise ValueError("pair cutoff radius must be finite and positive")
        if switch is not None and (
            not isfinite(switch) or switch < 0.0 or switch >= radius
        ):
            raise ValueError(
                "pair switch_start must satisfy 0 <= switch_start < radius"
            )
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "switch_start", switch)

    def to_payload(self) -> dict:
        return {
            "radius": float(self.radius).hex(),
            "switch_start": (
                None if self.switch_start is None else float(self.switch_start).hex()
            ),
        }


@dataclass(frozen=True)
class PairTopology:
    """Canonical undirected pair enumeration with explicit deterministic ownership."""

    atom_count: int
    pairs: tuple[tuple[int, int], ...]
    cutoff: PairCutoff | None = None
    ownership: str = PAIR_OWNERSHIP

    def __post_init__(self) -> None:
        if type(self.atom_count) is not int or self.atom_count < 0:
            raise ValueError("atom_count must be a nonnegative integer")
        pairs = tuple(tuple(pair) for pair in self.pairs)
        if any(
            len(pair) != 2
            or any(type(i) is not int for i in pair)
            or not (0 <= pair[0] < pair[1] < self.atom_count)
            for pair in pairs
        ):
            raise ValueError(
                "pairs must be canonical in-range undirected (i, j) with i < j"
            )
        if tuple(sorted(pairs)) != pairs or len(set(pairs)) != len(pairs):
            raise ValueError(
                "pair enumeration must be unique and lexicographically ordered"
            )
        if self.ownership != PAIR_OWNERSHIP:
            raise ValueError("unsupported pair ownership contract")
        if self.cutoff is not None and not isinstance(self.cutoff, PairCutoff):
            raise TypeError("cutoff must be a PairCutoff")
        object.__setattr__(self, "pairs", pairs)

    @classmethod
    def complete(cls, atom_count: int, *, cutoff: PairCutoff | None = None):
        pairs = tuple(
            (i, j) for i in range(atom_count) for j in range(i + 1, atom_count)
        )
        return cls(atom_count, pairs, cutoff=cutoff)

    @property
    def owners(self) -> tuple[int, ...]:
        return tuple(i for i, _ in self.pairs)

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "atom_count": self.atom_count,
            "pairs": [list(pair) for pair in self.pairs],
            "ownership": self.ownership,
            "owners": list(self.owners),
            "cutoff": None if self.cutoff is None else self.cutoff.to_payload(),
        }


@dataclass(frozen=True)
class GeometryIR:
    """Immutable atom metadata plus one differentiable Cartesian coordinate input."""

    elements: tuple[int, ...]
    parameter_identity: str = "none"
    coordinate_name: str = "coordinates"
    dtype: str = "float64"

    def __post_init__(self) -> None:
        elements = tuple(self.elements)
        if any(type(z) is not int or not 1 <= z <= 118 for z in elements):
            raise ValueError("elements must contain atomic numbers in [1, 118]")
        if not isinstance(self.parameter_identity, str) or not self.parameter_identity:
            raise ValueError("parameter_identity must be a nonempty string")
        if (
            not isinstance(self.coordinate_name, str)
            or not self.coordinate_name.isidentifier()
        ):
            raise ValueError("coordinate_name must be an identifier")
        if self.dtype != "float64":
            raise ValueError("GeometryIR is currently qualified for float64")
        object.__setattr__(self, "elements", elements)

    @property
    def atom_count(self) -> int:
        return len(self.elements)

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "elements": list(self.elements),
            "parameter_identity": self.parameter_identity,
            "coordinate_name": self.coordinate_name,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class PairTensorContext:
    """TensorIR nodes for one fixed pair topology."""

    geometry: GeometryIR
    topology: PairTopology
    coordinates: Node
    left: Node
    right: Node
    displacement: Node
    squared_distance: Node
    distance: Node
    atom_index: Index
    pair_index: Index
    cartesian_index: Index

    @property
    def pair_elements(self) -> tuple[tuple[int, int], ...]:
        z = self.geometry.elements
        return tuple((z[i], z[j]) for i, j in self.topology.pairs)


@dataclass(frozen=True)
class PairProgram:
    """Pair compiler contract around a shared TensorIR Program."""

    geometry: GeometryIR
    topology: PairTopology
    program: Program
    parameter_identity: str
    pair_kind: str
    lowering_version: int = LOWERING_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, GeometryIR):
            raise TypeError("geometry must be GeometryIR")
        if not isinstance(self.topology, PairTopology):
            raise TypeError("topology must be PairTopology")
        if not isinstance(self.program, Program):
            raise TypeError("program must be TensorIR Program")
        if self.geometry.atom_count != self.topology.atom_count:
            raise ValueError("geometry/topology atom counts disagree")
        if not isinstance(self.parameter_identity, str) or not self.parameter_identity:
            raise ValueError("parameter_identity must be nonempty")
        if not isinstance(self.pair_kind, str) or not self.pair_kind:
            raise ValueError("pair_kind must be nonempty")
        if self.lowering_version != LOWERING_VERSION:
            raise ValueError("unsupported pair lowering version")

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
            "pair_kind": self.pair_kind,
        }

    def validate_execution_identity(self, identity: str) -> None:
        if identity != self.identity:
            raise ValueError("stale pair compiler execution state")

    def coordinate_jvp(self):
        energy = self.program.outputs.get("energy")
        if energy is None or energy.spec.shape != ():
            raise ValueError("coordinate JVP requires scalar output named energy")
        return linearize(
            self.program, [self.geometry.coordinate_name], outputs=["energy"]
        )

    def coordinate_vjp(self):
        energy = self.program.outputs.get("energy")
        if energy is None or energy.spec.shape != ():
            raise ValueError("coordinate VJP requires scalar output named energy")
        return transpose_program(
            self.program, ["energy"], inputs=[self.geometry.coordinate_name]
        )


def lower_geometry(geometry: GeometryIR, topology: PairTopology) -> PairTensorContext:
    """Lower a fixed pair topology to gather/reshape/arithmetic TensorIR."""

    if not isinstance(geometry, GeometryIR) or not isinstance(topology, PairTopology):
        raise TypeError("lower_geometry requires GeometryIR and PairTopology")
    if geometry.atom_count != topology.atom_count:
        raise ValueError("geometry/topology atom counts disagree")
    atom_space = IndexSpace("atom", "atom", geometry.atom_count)
    pair_space = IndexSpace("pair", "pair", len(topology.pairs))
    cart_space = IndexSpace("cartesian", "cartesian", 3)
    atom = Index("a", atom_space)
    pair = Index("p", pair_space)
    cart = Index("c", cart_space)
    coordinates = input_tensor(
        geometry.coordinate_name,
        TensorSpec(
            (atom, cart), dtype=geometry.dtype, role="input", differentiable=True
        ),
    )
    left_positions = tuple(i for i, _ in topology.pairs)
    right_positions = tuple(j for _, j in topology.pairs)
    left = reshape(gather(coordinates, 0, left_positions), (pair, cart))
    right = reshape(gather(coordinates, 0, right_positions), (pair, cart))
    displacement = add(right, left, coefficients=(1, -1))
    squared_distance = reduce_sum(multiply(displacement, displacement), (1,))
    distance = sqrt(squared_distance)
    return PairTensorContext(
        geometry,
        topology,
        coordinates,
        left,
        right,
        displacement,
        squared_distance,
        distance,
        atom,
        pair,
        cart,
    )


def pair_to_system(value: Node, context: PairTensorContext) -> Node:
    """Reduce a per-pair scalar to one system scalar."""

    if value.spec.indices != (context.pair_index,):
        raise ValueError("pair_to_system requires one scalar value per canonical pair")
    return reduce_sum(value, (0,))


def pair_to_atom(
    value: Node, context: PairTensorContext, *, owners_only: bool = False
) -> Node:
    """Accumulate one scalar per pair onto incident atoms or deterministic owners."""

    if value.spec.indices != (context.pair_index,):
        raise ValueError("pair_to_atom requires one scalar value per canonical pair")
    atom, pair = context.atom_index, context.pair_index
    spec = TensorSpec(
        (Index("atom_row", atom.space), Index("pair_col", pair.space)),
        dtype=value.spec.dtype,
        role="constant",
    )
    rows = []
    for a in range(context.topology.atom_count):
        for owner, (i, j) in zip(context.topology.owners, context.topology.pairs):
            rows.append(int(a == owner) if owners_only else int(a == i or a == j))
    incidence = constant(tuple(rows), spec)
    return einsum("p,ap->a", value, incidence)


def build_pair_program(
    context: PairTensorContext,
    pair_energy: Node,
    *,
    parameter_identity: str,
    pair_kind: str,
) -> PairProgram:
    """Wrap a per-pair scalar equation as a system-energy compiler contract."""

    energy = pair_to_system(pair_energy, context)
    provenance = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "lowering_version": LOWERING_VERSION,
        "geometry": context.geometry.to_payload(),
        "topology": context.topology.to_payload(),
        "parameter_identity": parameter_identity,
        "pair_kind": pair_kind,
    }
    program = Program({"energy": energy}, provenance=provenance)
    return PairProgram(
        context.geometry,
        context.topology,
        program,
        parameter_identity,
        pair_kind,
    )


def inverse_power_program(
    geometry: GeometryIR,
    topology: PairTopology,
    coefficients,
    *,
    exponent=-1,
    parameter_identity: str | None = None,
) -> PairProgram:
    """Qualification potential: E = sum_p c_p * r_p**exponent."""

    context = lower_geometry(geometry, topology)
    if isinstance(coefficients, (int, str, Fraction)):
        coefficients = (coefficients,) * len(topology.pairs)
    else:
        coefficients = tuple(coefficients)
    if len(coefficients) != len(topology.pairs):
        raise ValueError("one coefficient is required per canonical pair")
    exact = tuple(rational(value) for value in coefficients)
    if parameter_identity is None:
        parameter_identity = canonical_hash(
            {
                "kind": "inverse_power",
                "coefficients": exact,
                "exponent": rational(exponent),
            }
        )
    coefficient = constant(
        coefficients,
        TensorSpec((context.pair_index,), dtype=geometry.dtype, role="constant"),
    )
    pair_energy = multiply(coefficient, power(context.distance, exponent))
    return build_pair_program(
        context,
        pair_energy,
        parameter_identity=parameter_identity,
        pair_kind="inverse_power",
    )
