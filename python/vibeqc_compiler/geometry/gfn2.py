"""Generated GFN2 short-range geometry equations for compiler qualification (#504).

The numerical parameter subset is derived from xTBloom commit 2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3,
whose GFN2 export is pinned to tblite. Production VibeQC does not import or
call xTBloom at runtime.

Rationale: .agents/notes/implemented/numerics/2026-09-20-gfn2-geometry-compiler-contract.md
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    constant,
    divide,
    exp,
    multiply,
    power,
    scatter_add,
    sqrt,
    transpose_program,
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

GFN2_SHORT_RANGE_VERSION = "gfn2-short-range-ir-v1"
GFN2_XTBLOOM_REVISION = "2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3"
GFN2_PARAMETER_JSON_SHA256 = (
    "de0f20e90b592b7b92f107eb672bd3dd29c1096f904d7a472b05693f9238ed1a"
)
GFN2_TBLITE_REVISION = "fa8a4416e8fe093d0075bc10ac875494c2a449a9"
GFN2_CUTOFF_BOHR = 25.0
GFN2_MINIMUM_CN_DISTANCE_SQUARED = 1.0e-12
GFN2_ANGSTROM_TO_BOHR = 1.8897261246204404
GFN2_CN_RADIUS_SCALE = (4.0 / 3.0) * GFN2_ANGSTROM_TO_BOHR
GFN2_CN_FIRST_STEEPNESS = 10.0
GFN2_CN_SECOND_STEEPNESS = 20.0
GFN2_CN_SECOND_RADIUS_SHIFT_BOHR = 2.0
GFN2_REPULSION_KEXP = 1.5
GFN2_REPULSION_KLIGHT = 1.0


@dataclass(frozen=True)
class Gfn2ShortRangeElement:
    atomic_number: int
    covalent_radius_angstrom: float
    arep: float
    zeff: float

    @property
    def covalent_radius_bohr(self) -> float:
        return GFN2_CN_RADIUS_SCALE * self.covalent_radius_angstrom


_ELEMENT_ROWS = (
    (1, 0.32, 2.213717, 1.105388),
    (2, 0.46, 3.60467, 1.094283),
    (3, 1.2, 0.475307, 1.289367),
    (4, 0.94, 0.939696, 4.221216),
    (5, 0.77, 1.373856, 7.192431),
    (6, 0.75, 1.247655, 4.231078),
    (7, 0.71, 1.682689, 5.242592),
    (8, 0.63, 2.165712, 5.784415),
    (9, 0.64, 2.421394, 7.021486),
    (10, 0.67, 3.318479, 11.041068),
    (11, 1.4, 0.572728, 5.244917),
    (12, 1.25, 0.917975, 18.083164),
    (13, 1.13, 0.876623, 17.867328),
    (14, 1.04, 1.187323, 40.001111),
    (15, 1.1, 1.143343, 19.683502),
    (16, 1.02, 1.214553, 14.99509),
    (17, 0.99, 1.577144, 17.353134),
    (18, 0.96, 0.896198, 7.266606),
    (19, 1.76, 0.482206, 10.439482),
    (20, 1.54, 0.683051, 14.786701),
    (21, 1.33, 0.574299, 8.004267),
    (22, 1.22, 0.723104, 12.036336),
    (23, 1.21, 0.928532, 15.677873),
    (24, 1.1, 0.966993, 19.517914),
    (25, 1.07, 1.0711, 18.760605),
    (26, 1.04, 1.113422, 20.360089),
    (27, 1.0, 1.241717, 27.127744),
    (28, 0.99, 1.077516, 10.533269),
    (29, 1.01, 0.998768, 9.913846),
    (30, 1.09, 1.160262, 22.099503),
    (31, 1.12, 1.122923, 31.14675),
    (32, 1.09, 1.222349, 42.100144),
    (33, 1.15, 1.249372, 39.147587),
    (34, 1.1, 1.230284, 27.426779),
    (35, 1.14, 1.296174, 32.845361),
    (36, 1.17, 0.908074, 17.363803),
    (37, 1.89, 0.574054, 44.338211),
    (38, 1.67, 0.697345, 34.365525),
    (39, 1.47, 0.706172, 17.326237),
    (40, 1.39, 0.681106, 24.263093),
    (41, 1.32, 0.865552, 30.562732),
    (42, 1.24, 1.034519, 48.312796),
    (43, 1.15, 1.019565, 44.779882),
    (44, 1.13, 1.031669, 28.070247),
    (45, 1.13, 1.094599, 38.035941),
    (46, 1.08, 1.092745, 28.6747),
    (47, 1.15, 0.678344, 6.493286),
    (48, 1.23, 0.936236, 26.226628),
    (49, 1.28, 1.024007, 63.85424),
    (50, 1.26, 1.139959, 80.053438),
    (51, 1.26, 1.122937, 77.05756),
    (52, 1.23, 1.000712, 48.614745),
    (53, 1.32, 1.017946, 63.319176),
    (54, 1.31, 1.012036, 51.188398),
    (55, 2.09, 0.585257, 67.249039),
    (56, 1.76, 0.716259, 46.984607),
    (57, 1.62, 0.737643, 50.927529),
    (58, 1.47, 0.72995, 48.676714),
    (59, 1.58, 0.734624, 47.669448),
    (60, 1.57, 0.739299, 46.662183),
    (61, 1.56, 0.743973, 45.654917),
    (62, 1.55, 0.748648, 44.647651),
    (63, 1.51, 0.753322, 43.640385),
    (64, 1.52, 0.757996, 42.63312),
    (65, 1.51, 0.762671, 41.625854),
    (66, 1.5, 0.767345, 40.618588),
    (67, 1.49, 0.77202, 39.611322),
    (68, 1.49, 0.776694, 38.604057),
    (69, 1.48, 0.781368, 37.596791),
    (70, 1.53, 0.786043, 36.589525),
    (71, 1.46, 0.790717, 35.582259),
    (72, 1.37, 0.852852, 40.186772),
    (73, 1.31, 0.990234, 54.666156),
    (74, 1.23, 1.018805, 55.899801),
    (75, 1.18, 1.170412, 80.410086),
    (76, 1.16, 1.221937, 62.809871),
    (77, 1.11, 1.197148, 56.045639),
    (78, 1.12, 1.204081, 53.881425),
    (79, 1.13, 0.91921, 14.711475),
    (80, 1.32, 1.13736, 51.577544),
    (81, 1.3, 1.399312, 58.801614),
    (82, 1.3, 1.179922, 102.368258),
    (83, 1.36, 1.13086, 132.896832),
    (84, 1.31, 0.957939, 52.301232),
    (85, 1.38, 0.963878, 81.771063),
    (86, 1.42, 0.965577, 128.13358),
)

GFN2_SHORT_RANGE_ELEMENTS = MappingProxyType(
    {
        z: Gfn2ShortRangeElement(z, radius, arep, zeff)
        for z, radius, arep, zeff in _ELEMENT_ROWS
    }
)

GFN2_SHORT_RANGE_PARAMETER_IDENTITY = canonical_hash(
    {
        "version": GFN2_SHORT_RANGE_VERSION,
        "xtbloom_revision": GFN2_XTBLOOM_REVISION,
        "gfn2_parameter_json_sha256": GFN2_PARAMETER_JSON_SHA256,
        "tblite_revision": GFN2_TBLITE_REVISION,
        "coordination": {
            "mctc": "0.5.2",
            "angstrom_to_bohr": float(GFN2_ANGSTROM_TO_BOHR).hex(),
            "radius_scale": float(GFN2_CN_RADIUS_SCALE).hex(),
            "first_steepness": float(GFN2_CN_FIRST_STEEPNESS).hex(),
            "second_steepness": float(GFN2_CN_SECOND_STEEPNESS).hex(),
            "second_radius_shift_bohr": float(GFN2_CN_SECOND_RADIUS_SHIFT_BOHR).hex(),
            "cutoff_bohr": float(GFN2_CUTOFF_BOHR).hex(),
        },
        "repulsion": {
            "kexp": float(GFN2_REPULSION_KEXP).hex(),
            "klight": float(GFN2_REPULSION_KLIGHT).hex(),
        },
        "elements": [
            {
                "z": item.atomic_number,
                "radius_angstrom": float(item.covalent_radius_angstrom).hex(),
                "arep": float(item.arep).hex(),
                "zeff": float(item.zeff).hex(),
            }
            for item in GFN2_SHORT_RANGE_ELEMENTS.values()
        ],
    }
)


def gfn2_element_parameters(atomic_number: int) -> Gfn2ShortRangeElement:
    if type(atomic_number) is not int:
        raise TypeError("GFN2 atomic number must be an integer")
    try:
        return GFN2_SHORT_RANGE_ELEMENTS[atomic_number]
    except KeyError as error:
        raise ValueError(
            "GFN2 short-range parameters support atomic numbers 1..86"
        ) from error


def gfn2_geometry(
    elements: Iterable[int], *, coordinate_name: str = "coordinates"
) -> GeometryIR:
    elements = tuple(elements)
    for atomic_number in elements:
        gfn2_element_parameters(atomic_number)
    return GeometryIR(
        elements,
        parameter_identity=GFN2_SHORT_RANGE_PARAMETER_IDENTITY,
        coordinate_name=coordinate_name,
    )


def _validated_gfn2_coordinates(
    geometry: GeometryIR, coordinates: ArrayLike
) -> np.ndarray:
    if not isinstance(geometry, GeometryIR):
        raise TypeError("geometry must be GeometryIR")
    coordinates = np.asarray(coordinates)
    if coordinates.dtype.kind not in "iuf":
        raise ValueError("GFN2 coordinates must be real numeric arrays")
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (geometry.atom_count, 3):
        raise ValueError("GFN2 coordinates must have shape (atom_count, 3)")
    if not np.isfinite(coordinates).all():
        raise ValueError("GFN2 coordinates must be finite")
    return coordinates


def _normalized_gfn2_system_atom_offsets(
    system_atom_offsets: Iterable[int], atom_count: int
) -> tuple[int, ...]:
    offsets = tuple(system_atom_offsets)
    if (
        len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != atom_count
        or any(type(value) is not int for value in offsets)
        or any(left >= right for left, right in pairwise(offsets))
    ):
        raise ValueError(
            "GFN2 batch atom offsets must be a strictly increasing partition "
            "from zero through atom_count"
        )
    return offsets


def _gfn2_pairs_for_ranges(
    coordinates: np.ndarray, ranges: Iterable[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    cutoff2 = GFN2_CUTOFF_BOHR * GFN2_CUTOFF_BOHR
    pairs = []
    for begin, end in ranges:
        for first in range(begin, end):
            for second in range(first + 1, end):
                displacement = coordinates[second] - coordinates[first]
                distance2 = float(np.dot(displacement, displacement))
                if distance2 < GFN2_MINIMUM_CN_DISTANCE_SQUARED:
                    raise ValueError(
                        "GFN2 coordination is undefined for coincident or "
                        "near-coincident atoms within one system"
                    )
                if distance2 <= cutoff2:
                    pairs.append((first, second))
    return tuple(pairs)


def build_gfn2_pair_topology(
    geometry: GeometryIR,
    coordinates: ArrayLike,
) -> PairTopology:
    """Build the exact molecular 25-bohr pair set for one geometry snapshot."""

    coordinates = _validated_gfn2_coordinates(geometry, coordinates)
    pairs = _gfn2_pairs_for_ranges(coordinates, ((0, geometry.atom_count),))
    return PairTopology(
        geometry.atom_count,
        pairs,
        cutoff=PairCutoff(GFN2_CUTOFF_BOHR),
    )


def build_gfn2_batch_pair_topology(
    geometry: GeometryIR,
    system_atom_offsets: Iterable[int],
    coordinates: ArrayLike,
) -> PairTopology:
    """Build one canonical pair topology for a heterogeneous ragged batch."""

    coordinates = _validated_gfn2_coordinates(geometry, coordinates)
    offsets = _normalized_gfn2_system_atom_offsets(
        system_atom_offsets, geometry.atom_count
    )
    pairs = _gfn2_pairs_for_ranges(coordinates, pairwise(offsets))
    return PairTopology(
        geometry.atom_count,
        pairs,
        cutoff=PairCutoff(GFN2_CUTOFF_BOHR),
    )


def _require_gfn2_topology(geometry: GeometryIR, topology: PairTopology) -> None:
    if geometry.atom_count != topology.atom_count:
        raise ValueError("GFN2 geometry/topology atom counts disagree")
    if (
        topology.cutoff is None
        or topology.cutoff.radius != GFN2_CUTOFF_BOHR
        or topology.cutoff.switch_start is not None
    ):
        raise ValueError("GFN2 short-range topology requires the sharp 25-bohr cutoff")


def _gfn2_pair_system_owners(
    topology: PairTopology, system_atom_offsets: tuple[int, ...]
) -> tuple[int, ...]:
    owners = []
    system = 0
    for first, second in topology.pairs:
        while first >= system_atom_offsets[system + 1]:
            system += 1
        if not (
            system_atom_offsets[system]
            <= first
            < second
            < system_atom_offsets[system + 1]
        ):
            raise ValueError("GFN2 batch topology contains a cross-system pair")
        owners.append(system)
    return tuple(owners)


def _pair_constant(context: PairTensorContext, values: Iterable[float]) -> Node:
    literals = tuple(repr(float(value)) for value in values)
    if len(literals) != len(context.topology.pairs):
        raise ValueError("GFN2 pair parameter length disagrees with topology")
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


@dataclass(frozen=True)
class Gfn2ShortRangeProgram:
    """Geometry-owned GFN2 CN/repulsion graph with no MethodIR dependency."""

    geometry: GeometryIR
    topology: PairTopology
    program: Program
    parameter_identity: str = GFN2_SHORT_RANGE_PARAMETER_IDENTITY
    version: str = GFN2_SHORT_RANGE_VERSION

    def __post_init__(self) -> None:
        if self.geometry.parameter_identity != self.parameter_identity:
            raise ValueError("GFN2 geometry parameter identity mismatch")
        _require_gfn2_topology(self.geometry, self.topology)
        if self.version != GFN2_SHORT_RANGE_VERSION:
            raise ValueError("unsupported GFN2 short-range compiler version")

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
            raise ValueError("stale GFN2 geometry compiler execution state")

    def validate_coordinates(self, coordinates: ArrayLike) -> None:
        expected = build_gfn2_pair_topology(self.geometry, coordinates)
        if expected.identity != self.topology.identity:
            raise ValueError("stale GFN2 pair topology for changed coordinates")

    def coordinate_vjp(self, output: str) -> VJPProgram:
        if output not in ("coordination", "repulsion_energy"):
            raise ValueError("unknown GFN2 short-range derivative output")
        return transpose_program(
            self.program,
            [output],
            inputs=[self.geometry.coordinate_name],
        )


def _gfn2_pair_terms(context: PairTensorContext) -> tuple[Node, Node]:
    """Build CN and per-pair repulsion once for scalar and ragged consumers."""

    pair_elements = context.pair_elements
    parameters = [
        (
            gfn2_element_parameters(first),
            gfn2_element_parameters(second),
        )
        for first, second in pair_elements
    ]

    pair_count = len(context.topology.pairs)
    one = _pair_constant(context, (1.0,) * pair_count)
    minus_one = _pair_constant(context, (-1.0,) * pair_count)
    inverse_distance = power(context.distance, -1)

    radii = _pair_constant(
        context,
        (
            first.covalent_radius_bohr + second.covalent_radius_bohr
            for first, second in parameters
        ),
    )
    shifted_radii = _pair_constant(
        context,
        (
            first.covalent_radius_bohr
            + second.covalent_radius_bohr
            + GFN2_CN_SECOND_RADIUS_SHIFT_BOHR
            for first, second in parameters
        ),
    )
    first_ratio = multiply(radii, inverse_distance)
    second_ratio = multiply(shifted_radii, inverse_distance)
    first_delta = add(first_ratio, one, coefficients=(1, -1))
    second_delta = add(second_ratio, one, coefficients=(1, -1))
    first_argument = multiply(
        _pair_constant(context, (GFN2_CN_FIRST_STEEPNESS,) * pair_count),
        first_delta,
    )
    second_argument = multiply(
        _pair_constant(context, (GFN2_CN_SECOND_STEEPNESS,) * pair_count),
        second_delta,
    )
    pair_coordination = multiply(
        _logistic(first_argument, one, minus_one),
        _logistic(second_argument, one, minus_one),
    )
    coordination = pair_to_atom(pair_coordination, context)

    light = tuple(
        float(first.atomic_number <= 2 and second.atomic_number <= 2)
        for first, second in parameters
    )
    heavy = tuple(1.0 - value for value in light)
    heavy_distance = multiply(context.distance, sqrt(context.distance))
    distance_power = add(
        multiply(_pair_constant(context, light), context.distance),
        multiply(_pair_constant(context, heavy), heavy_distance),
    )
    pair_alpha = _pair_constant(
        context,
        ((first.arep * second.arep) ** 0.5 for first, second in parameters),
    )
    pair_charge = _pair_constant(
        context,
        (first.zeff * second.zeff for first, second in parameters),
    )
    decay_argument = multiply(minus_one, multiply(pair_alpha, distance_power))
    pair_repulsion = multiply(
        multiply(pair_charge, exp(decay_argument)),
        inverse_distance,
    )
    return coordination, pair_repulsion


@dataclass(frozen=True)
class Gfn2ShortRangeBatchProgram:
    """One TensorIR graph for a fixed heterogeneous ragged molecular batch."""

    geometry: GeometryIR
    system_atom_offsets: tuple[int, ...]
    topology: PairTopology
    program: Program
    parameter_identity: str = GFN2_SHORT_RANGE_PARAMETER_IDENTITY
    version: str = GFN2_SHORT_RANGE_VERSION

    def __post_init__(self) -> None:
        if self.geometry.parameter_identity != self.parameter_identity:
            raise ValueError("GFN2 geometry parameter identity mismatch")
        offsets = _normalized_gfn2_system_atom_offsets(
            self.system_atom_offsets, self.geometry.atom_count
        )
        object.__setattr__(self, "system_atom_offsets", offsets)
        _require_gfn2_topology(self.geometry, self.topology)
        _gfn2_pair_system_owners(self.topology, offsets)
        if self.version != GFN2_SHORT_RANGE_VERSION:
            raise ValueError("unsupported GFN2 short-range compiler version")

    @property
    def identity(self) -> str:
        return canonical_hash(
            {
                "version": self.version,
                "geometry": self.geometry.to_payload(),
                "system_atom_offsets": list(self.system_atom_offsets),
                "topology": self.topology.to_payload(),
                "equation": self.program.logical_hash,
                "parameter_identity": self.parameter_identity,
            }
        )

    def validate_execution_identity(self, identity: str) -> None:
        if identity != self.identity:
            raise ValueError("stale GFN2 batch compiler execution state")

    def validate_coordinates(self, coordinates: ArrayLike) -> None:
        expected = build_gfn2_batch_pair_topology(
            self.geometry, self.system_atom_offsets, coordinates
        )
        if expected.identity != self.topology.identity:
            raise ValueError("stale GFN2 batch pair topology for changed coordinates")

    def coordinate_vjp(self, output: str) -> VJPProgram:
        if output not in ("coordination", "repulsion_energy"):
            raise ValueError("unknown GFN2 short-range derivative output")
        return transpose_program(
            self.program,
            [output],
            inputs=[self.geometry.coordinate_name],
        )


def build_gfn2_short_range_program(
    geometry: GeometryIR,
    topology: PairTopology,
) -> Gfn2ShortRangeProgram:
    """Compile one-system GFN2 CN and nuclear repulsion through TensorIR."""

    if not isinstance(geometry, GeometryIR):
        raise TypeError("geometry must be GeometryIR")
    if not isinstance(topology, PairTopology):
        raise TypeError("topology must be PairTopology")
    if geometry.parameter_identity != GFN2_SHORT_RANGE_PARAMETER_IDENTITY:
        raise ValueError(
            "geometry is not bound to the GFN2 short-range parameter identity"
        )
    for atomic_number in geometry.elements:
        gfn2_element_parameters(atomic_number)
    _require_gfn2_topology(geometry, topology)

    context = lower_geometry(geometry, topology)
    coordination, pair_repulsion = _gfn2_pair_terms(context)
    repulsion_energy = pair_to_system(pair_repulsion, context)
    program = Program(
        {
            "coordination": coordination,
            "repulsion_energy": repulsion_energy,
        },
        provenance={
            "kind": "gfn2-short-range",
            "version": GFN2_SHORT_RANGE_VERSION,
            "parameter_identity": GFN2_SHORT_RANGE_PARAMETER_IDENTITY,
            "xtbloom_revision": GFN2_XTBLOOM_REVISION,
            "topology": topology.to_payload(),
        },
    )
    return Gfn2ShortRangeProgram(geometry, topology, program)


def build_gfn2_short_range_batch_program(
    geometry: GeometryIR,
    system_atom_offsets: Iterable[int],
    topology: PairTopology,
) -> Gfn2ShortRangeBatchProgram:
    """Compile a heterogeneous ragged GFN2 CN/repulsion batch as one graph."""

    if not isinstance(geometry, GeometryIR):
        raise TypeError("geometry must be GeometryIR")
    if not isinstance(topology, PairTopology):
        raise TypeError("topology must be PairTopology")
    if geometry.parameter_identity != GFN2_SHORT_RANGE_PARAMETER_IDENTITY:
        raise ValueError(
            "geometry is not bound to the GFN2 short-range parameter identity"
        )
    for atomic_number in geometry.elements:
        gfn2_element_parameters(atomic_number)
    offsets = _normalized_gfn2_system_atom_offsets(
        system_atom_offsets, geometry.atom_count
    )
    _require_gfn2_topology(geometry, topology)
    pair_system_owners = _gfn2_pair_system_owners(topology, offsets)

    context = lower_geometry(geometry, topology)
    coordination, pair_repulsion = _gfn2_pair_terms(context)
    system_index = Index(
        "s",
        IndexSpace("system", "batch", len(offsets) - 1),
    )
    repulsion_energy = scatter_add(
        pair_repulsion,
        0,
        pair_system_owners,
        system_index,
    )
    program = Program(
        {
            "coordination": coordination,
            "repulsion_energy": repulsion_energy,
        },
        provenance={
            "kind": "gfn2-short-range-ragged-batch",
            "version": GFN2_SHORT_RANGE_VERSION,
            "parameter_identity": GFN2_SHORT_RANGE_PARAMETER_IDENTITY,
            "xtbloom_revision": GFN2_XTBLOOM_REVISION,
            "system_atom_offsets": list(offsets),
            "topology": topology.to_payload(),
        },
    )
    return Gfn2ShortRangeBatchProgram(
        geometry,
        offsets,
        topology,
        program,
    )
