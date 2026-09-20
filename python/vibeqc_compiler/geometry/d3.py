"""D3(BJ) geometry/pair lowering through the shared TensorIR.

This module owns the compiler representation of the two-body D3(BJ) equation.
The canonical parameter tables remain the pinned repository assets used by the
native reference/runtime; no second scientific table copy is maintained here.
"""

from __future__ import annotations

import functools
import hashlib
import math
import typing
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    VJPProgram,
    add,
    constant,
    divide,
    exp,
    gather,
    multiply,
    power,
    reshape,
    scatter_add,
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

D3_TABLE_SHA256 = "9ff932ea598f690c1fb599a67762060ba1907102d5ec132164f2a7e8886cd22e"
D3_RADII_SHA256 = "92b32fada844a337204b84f2d961473bad5737240765eb8d0727a62827de5111"
D3_COMPILER_VERSION = "d3-bj-geometry-pair-ir-v1"
D3_MINIMUM_DISTANCE_SQUARED = 1.0e-12
D3_REFERENCE_SLOTS = 7


class D3SpecLike(typing.Protocol):
    """Structural boundary for MethodIR's D3Spec without geometry -> method imports."""

    s6: float
    s8: float
    a1: float
    a2: float
    table_sha256: str
    radii_sha256: str
    s9: float
    damping: str
    cn_cutoff: float | None
    pair_cutoff: float | None
    pair_switch_width: float
    version: str


@dataclass(frozen=True)
class D3CompilerSpec:
    """Compiler-owned immutable copy of the supported D3(BJ) specification."""

    s6: float
    s8: float
    a1: float
    a2: float
    table_sha256: str
    radii_sha256: str
    s9: float = 0.0
    damping: str = "bj"
    cn_cutoff: float | None = None
    pair_cutoff: float | None = None
    pair_switch_width: float = 0.0
    version: str = "d3-bj-spec-v1"

    @classmethod
    def from_spec(cls, spec: D3SpecLike | D3CompilerSpec) -> D3CompilerSpec:
        if isinstance(spec, cls):
            return spec
        fields = (
            "s6",
            "s8",
            "a1",
            "a2",
            "table_sha256",
            "radii_sha256",
            "s9",
            "damping",
            "cn_cutoff",
            "pair_cutoff",
            "pair_switch_width",
            "version",
        )
        missing = tuple(name for name in fields if not hasattr(spec, name))
        if missing:
            raise TypeError(f"D3 specification is missing fields {missing}")
        return cls(**{name: getattr(spec, name) for name in fields})

    def __post_init__(self) -> None:
        if self.version != "d3-bj-spec-v1" or self.damping != "bj":
            raise ValueError("compiler supports only versioned two-body D3(BJ)")
        for name in ("s6", "s8", "a1", "a2", "s9", "pair_switch_width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite real scalar")
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value if value else 0.0)
        if self.s9 != 0.0:
            raise ValueError("D3 ATM is not part of the two-body compiler lowering")
        if self.a1 == 0.0 and self.a2 == 0.0:
            raise ValueError("D3(BJ) requires a positive damping radius")
        for name in ("cn_cutoff", "pair_cutoff"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be None or a finite real scalar")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive when present")
            object.__setattr__(self, name, value)
        if self.pair_switch_width and (
            self.pair_cutoff is None or self.pair_switch_width >= self.pair_cutoff
        ):
            raise ValueError("D3 pair switch requires 0 < width < pair cutoff")
        if self.table_sha256 != D3_TABLE_SHA256:
            raise ValueError("unsupported D3 reference-table identity")
        if self.radii_sha256 != D3_RADII_SHA256:
            raise ValueError("unsupported D3 covalent-radii identity")

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict:
        return {
            "s6": self.s6,
            "s8": self.s8,
            "a1": self.a1,
            "a2": self.a2,
            "table_sha256": self.table_sha256,
            "radii_sha256": self.radii_sha256,
            "s9": self.s9,
            "damping": self.damping,
            "cn_cutoff": self.cn_cutoff,
            "pair_cutoff": self.pair_cutoff,
            "pair_switch_width": self.pair_switch_width,
            "version": self.version,
        }


@dataclass(frozen=True)
class _ElementRecord:
    reference_count: int
    reference_offset: int
    r4r2: float
    covalent_radius: float


@dataclass(frozen=True)
class _PairRecord:
    c6_offset: int
    first_reference_count: int
    second_reference_count: int


@dataclass(frozen=True)
class _D3Tables:
    elements: tuple[_ElementRecord, ...]
    pairs: tuple[_PairRecord, ...]
    reference_cn: tuple[float, ...]
    reference_c6: tuple[float, ...]


def _sha256(path: typing.Any) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@functools.lru_cache(maxsize=1)
def _d3_tables() -> _D3Tables:
    import json

    table_path = asset_path("external/xtbloom-d3/gfn1_d3.json")
    radii_path = asset_path("external/xtbloom-d3/covalent_radii.json")
    if _sha256(table_path) != D3_TABLE_SHA256:
        raise ValueError("pinned D3 table digest does not match compiler identity")
    if _sha256(radii_path) != D3_RADII_SHA256:
        raise ValueError("pinned D3 radii digest does not match compiler identity")
    raw = json.loads(table_path.read_text())
    radii = json.loads(radii_path.read_text())
    if (
        len(raw["elements"]) != 86
        or len(raw["pair_records"]) != 3741
        or len(raw["coordination_numbers"]) != 237
        or len(raw["c6"]) != 28455
        or len(raw["r4r2"]) != 86
        or len(radii) != 86
    ):
        raise ValueError("pinned D3 table shape changed")
    elements = tuple(
        _ElementRecord(
            int(record["reference_count"]),
            int(record["reference_offset"]),
            float(raw["r4r2"][index]),
            float(radii[index]),
        )
        for index, record in enumerate(raw["elements"])
    )
    pairs = tuple(
        _PairRecord(
            int(record["c6_offset"]),
            int(record["first_reference_count"]),
            int(record["second_reference_count"]),
        )
        for record in raw["pair_records"]
    )
    if any(not 1 <= item.reference_count <= D3_REFERENCE_SLOTS for item in elements):
        raise ValueError("unsupported D3 reference count")
    return _D3Tables(
        elements,
        pairs,
        tuple(float(value) for value in raw["coordination_numbers"]),
        tuple(float(value) for value in raw["c6"]),
    )


def _pair_record_index(first_z: int, second_z: int) -> int:
    low, high = sorted((first_z, second_z))
    return low - 1 + high * (high - 1) // 2


def _reference_c6(
    tables: _D3Tables,
    first_z: int,
    second_z: int,
    first_ref: int,
    second_ref: int,
) -> float:
    first_element = tables.elements[first_z - 1]
    second_element = tables.elements[second_z - 1]
    if (
        first_ref >= first_element.reference_count
        or second_ref >= second_element.reference_count
    ):
        return 0.0
    record = tables.pairs[_pair_record_index(first_z, second_z)]
    if first_z <= second_z:
        offset = (
            record.c6_offset + second_ref * record.first_reference_count + first_ref
        )
    else:
        offset = (
            record.c6_offset + first_ref * record.first_reference_count + second_ref
        )
    return tables.reference_c6[offset]


@dataclass(frozen=True)
class D3PairTopology:
    """PairIR topology plus D3 cutoff/switch regions for one geometry snapshot."""

    topology: PairTopology
    spec_identity: str
    cn_active: tuple[bool, ...]
    energy_regions: tuple[str, ...]

    def __post_init__(self) -> None:
        pair_count = len(self.topology.pairs)
        if len(self.cn_active) != pair_count or len(self.energy_regions) != pair_count:
            raise ValueError("D3 pair masks must match PairTopology")
        if any(
            region not in {"off", "inner", "switch"} for region in self.energy_regions
        ):
            raise ValueError("invalid D3 pair switch region")
        if not isinstance(self.spec_identity, str) or not self.spec_identity:
            raise ValueError("D3 topology requires a specification identity")

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict:
        return {
            "kind": "d3-bj-pair-topology",
            "version": D3_COMPILER_VERSION,
            "topology": self.topology.to_payload(),
            "spec_identity": self.spec_identity,
            "cn_active": list(self.cn_active),
            "energy_regions": list(self.energy_regions),
        }


def d3_geometry(
    elements: typing.Iterable[int],
    spec: D3SpecLike | D3CompilerSpec,
    *,
    coordinate_name: str = "coordinates",
) -> GeometryIR:
    """Bind supported D3 elements and parameters to GeometryIR."""

    compiler_spec = D3CompilerSpec.from_spec(spec)
    values = tuple(elements)
    if not values:
        raise ValueError("D3 requires at least one atom")
    if any(type(z) is not int or not 1 <= z <= 86 for z in values):
        raise ValueError("D3 compiler tables support atomic numbers 1..86")
    return GeometryIR(
        values,
        parameter_identity=compiler_spec.identity,
        coordinate_name=coordinate_name,
    )


def _energy_region(spec: D3CompilerSpec, distance: float) -> str:
    if spec.pair_cutoff is None:
        return "inner"
    if spec.pair_switch_width == 0.0:
        return "inner" if distance <= spec.pair_cutoff else "off"
    if distance >= spec.pair_cutoff:
        return "off"
    if distance <= spec.pair_cutoff - spec.pair_switch_width:
        return "inner"
    return "switch"


def _native_logistic(argument: float) -> float:
    value = math.exp(-abs(argument))
    return 1.0 / (1.0 + value) if argument >= 0.0 else value / (1.0 + value)


def _validate_reference_weight_domain(
    geometry: GeometryIR,
    state: D3PairTopology,
    coordinates: np.ndarray,
) -> None:
    tables = _d3_tables()
    coordination = np.zeros(geometry.atom_count, dtype=np.float64)
    for active, (first, second) in zip(state.cn_active, state.topology.pairs):
        if not active:
            continue
        displacement = coordinates[first] - coordinates[second]
        distance = math.sqrt(float(np.dot(displacement, displacement)))
        radius = (
            tables.elements[geometry.elements[first] - 1].covalent_radius
            + tables.elements[geometry.elements[second] - 1].covalent_radius
        )
        value = _native_logistic(16.0 * (radius / distance - 1.0))
        coordination[first] += value
        coordination[second] += value
    for atom, atomic_number in enumerate(geometry.elements):
        element = tables.elements[atomic_number - 1]
        norm = 0.0
        for ref in range(element.reference_count):
            reference = tables.reference_cn[element.reference_offset + ref]
            delta = reference - float(coordination[atom])
            norm += math.exp(-4.0 * delta * delta)
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError(
                "D3 reference-weight normalization requires native fallback "
                "for this geometry"
            )


def _normalized_d3_system_atom_offsets(
    system_atom_offsets: typing.Iterable[int], atom_count: int
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
            "D3 batch atom offsets must be a strictly increasing partition "
            "from zero through atom_count"
        )
    return offsets


def _build_d3_pair_topology_for_ranges(
    geometry: GeometryIR,
    compiler_spec: D3CompilerSpec,
    coordinates: np.ndarray,
    ranges: typing.Iterable[tuple[int, int]],
) -> D3PairTopology:
    pairs: list[tuple[int, int]] = []
    cn_active: list[bool] = []
    energy_regions: list[str] = []
    for begin, end in ranges:
        for first in range(begin, end):
            for second in range(first + 1, end):
                displacement = coordinates[first] - coordinates[second]
                distance_squared = float(np.dot(displacement, displacement))
                if (
                    not math.isfinite(distance_squared)
                    or distance_squared < D3_MINIMUM_DISTANCE_SQUARED
                ):
                    raise ValueError(
                        "D3 is undefined for coincident or near-coincident atoms"
                    )
                distance = math.sqrt(distance_squared)
                cn = (
                    compiler_spec.cn_cutoff is None
                    or distance_squared <= compiler_spec.cn_cutoff**2
                )
                region = _energy_region(compiler_spec, distance)
                if cn or region != "off":
                    pairs.append((first, second))
                    cn_active.append(cn)
                    energy_regions.append(region)

    union_cutoff = None
    if compiler_spec.cn_cutoff is not None and compiler_spec.pair_cutoff is not None:
        union_cutoff = PairCutoff(
            max(compiler_spec.cn_cutoff, compiler_spec.pair_cutoff)
        )
    state = D3PairTopology(
        PairTopology(
            geometry.atom_count,
            tuple(pairs),
            cutoff=union_cutoff,
        ),
        compiler_spec.identity,
        tuple(cn_active),
        tuple(energy_regions),
    )
    _validate_reference_weight_domain(geometry, state, coordinates)
    return state


def build_d3_pair_topology(
    geometry: GeometryIR,
    spec: D3SpecLike | D3CompilerSpec,
    coordinates: object,
) -> D3PairTopology:
    """Build the exact CN/energy pair-state contract for one geometry snapshot."""

    compiler_spec = D3CompilerSpec.from_spec(spec)
    if geometry.parameter_identity != compiler_spec.identity:
        raise ValueError(
            "D3 GeometryIR parameter identity does not match specification"
        )
    xyz = np.asarray(coordinates, dtype=np.float64)
    if xyz.shape != (geometry.atom_count, 3):
        raise ValueError("D3 coordinates must have shape (atom_count, 3)")
    if not np.isfinite(xyz).all():
        raise ValueError("D3 coordinates must be finite")
    return _build_d3_pair_topology_for_ranges(
        geometry,
        compiler_spec,
        xyz,
        ((0, geometry.atom_count),),
    )


def build_d3_batch_pair_topology(
    geometry: GeometryIR,
    spec: D3SpecLike | D3CompilerSpec,
    system_atom_offsets: typing.Iterable[int],
    coordinates: object,
) -> D3PairTopology:
    """Build one pair-state for a heterogeneous ragged molecular batch."""

    compiler_spec = D3CompilerSpec.from_spec(spec)
    if geometry.parameter_identity != compiler_spec.identity:
        raise ValueError(
            "D3 GeometryIR parameter identity does not match specification"
        )
    offsets = _normalized_d3_system_atom_offsets(
        system_atom_offsets, geometry.atom_count
    )
    xyz = np.asarray(coordinates, dtype=np.float64)
    if xyz.shape != (geometry.atom_count, 3):
        raise ValueError("D3 coordinates must have shape (atom_count, 3)")
    if not np.isfinite(xyz).all():
        raise ValueError("D3 coordinates must be finite")
    return _build_d3_pair_topology_for_ranges(
        geometry,
        compiler_spec,
        xyz,
        pairwise(offsets),
    )


def _d3_pair_system_owners(
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
            raise ValueError("D3 batch topology contains a cross-system pair")
        owners.append(system)
    return tuple(owners)


def _pair_constant(
    context: PairTensorContext,
    values: typing.Iterable[float],
) -> Node:
    encoded = tuple(repr(float(value)) for value in values)
    if len(encoded) != len(context.topology.pairs):
        raise ValueError("D3 pair constant length disagrees with topology")
    return constant(
        encoded,
        TensorSpec(
            (context.pair_index,),
            dtype=context.geometry.dtype,
            role="constant",
        ),
    )


def _atom_constant(node: Node, values: typing.Iterable[float]) -> Node:
    encoded = tuple(repr(float(value)) for value in values)
    if len(encoded) != node.spec.size:
        raise ValueError("D3 atom constant length disagrees with geometry")
    return constant(
        encoded,
        TensorSpec(
            node.spec.indices,
            dtype=node.spec.dtype,
            role="constant",
        ),
    )


def _atom_to_pair(
    value: Node,
    context: PairTensorContext,
    positions: tuple[int, ...],
) -> Node:
    return reshape(gather(value, 0, positions), (context.pair_index,))


def _coordination(
    context: PairTensorContext,
    state: D3PairTopology,
) -> Node:
    tables = _d3_tables()
    pair_count = len(context.topology.pairs)
    radii = _pair_constant(
        context,
        (
            tables.elements[first_z - 1].covalent_radius
            + tables.elements[second_z - 1].covalent_radius
            for first_z, second_z in context.pair_elements
        ),
    )
    one = _pair_constant(context, (1.0,) * pair_count)
    scale = _pair_constant(context, (16.0,) * pair_count)
    active = _pair_constant(
        context,
        (float(value) for value in state.cn_active),
    )
    ratio = divide(radii, context.distance)
    argument = multiply(
        scale,
        add(ratio, one, coefficients=(1, -1)),
    )
    minus_argument = multiply(
        _pair_constant(context, (-1.0,) * pair_count),
        argument,
    )
    pair_cn = multiply(
        active,
        divide(one, add(one, exp(minus_argument))),
    )
    return pair_to_atom(pair_cn, context)


def _reference_weights(
    geometry: GeometryIR,
    coordination: Node,
) -> tuple[Node, ...]:
    tables = _d3_tables()
    minus_four = _atom_constant(
        coordination,
        (-4.0,) * geometry.atom_count,
    )
    unnormalized = []
    for ref in range(D3_REFERENCE_SLOTS):
        references = []
        valid = []
        for atomic_number in geometry.elements:
            element = tables.elements[atomic_number - 1]
            is_valid = ref < element.reference_count
            valid.append(float(is_valid))
            references.append(
                tables.reference_cn[element.reference_offset + ref] if is_valid else 0.0
            )
        reference = _atom_constant(coordination, references)
        mask = _atom_constant(coordination, valid)
        delta = add(reference, coordination, coefficients=(1, -1))
        exponent = multiply(
            minus_four,
            multiply(delta, delta),
        )
        unnormalized.append(multiply(mask, exp(exponent)))
    norm = add(*unnormalized)
    return tuple(divide(value, norm) for value in unnormalized)


def _interpolated_c6(
    context: PairTensorContext,
    weights: tuple[Node, ...],
) -> Node:
    tables = _d3_tables()
    left_positions = tuple(first for first, _ in context.topology.pairs)
    right_positions = tuple(second for _, second in context.topology.pairs)
    left = tuple(_atom_to_pair(value, context, left_positions) for value in weights)
    right = tuple(_atom_to_pair(value, context, right_positions) for value in weights)
    terms = []
    for first_ref in range(D3_REFERENCE_SLOTS):
        for second_ref in range(D3_REFERENCE_SLOTS):
            references = _pair_constant(
                context,
                (
                    _reference_c6(
                        tables,
                        first_z,
                        second_z,
                        first_ref,
                        second_ref,
                    )
                    for first_z, second_z in context.pair_elements
                ),
            )
            terms.append(
                multiply(
                    multiply(
                        left[first_ref],
                        right[second_ref],
                    ),
                    references,
                )
            )
    return add(*terms)


def _switch(
    context: PairTensorContext,
    state: D3PairTopology,
    spec: D3CompilerSpec,
) -> Node:
    pair_count = len(context.topology.pairs)
    inner = _pair_constant(
        context,
        (1.0 if region == "inner" else 0.0 for region in state.energy_regions),
    )
    if spec.pair_switch_width == 0.0:
        return inner
    pair_cutoff = spec.pair_cutoff
    if pair_cutoff is None:
        raise ValueError("D3 pair switch requires a finite pair cutoff")
    switching = _pair_constant(
        context,
        (1.0 if region == "switch" else 0.0 for region in state.energy_regions),
    )
    cutoff = _pair_constant(
        context,
        (pair_cutoff,) * pair_count,
    )
    width = _pair_constant(
        context,
        (spec.pair_switch_width,) * pair_count,
    )
    x = divide(
        add(cutoff, context.distance, coefficients=(1, -1)),
        width,
    )
    x2 = multiply(x, x)
    x3 = multiply(x2, x)
    six_x = multiply(
        _pair_constant(context, (6.0,) * pair_count),
        x,
    )
    bracket = add(
        _pair_constant(context, (10.0,) * pair_count),
        multiply(
            x,
            add(
                _pair_constant(context, (-15.0,) * pair_count),
                six_x,
            ),
        ),
    )
    polynomial = multiply(x3, bracket)
    return add(inner, multiply(switching, polynomial))


def _pair_energy(
    context: PairTensorContext,
    state: D3PairTopology,
    spec: D3CompilerSpec,
    c6: Node,
) -> Node:
    tables = _d3_tables()
    pair_count = len(context.topology.pairs)
    rr_values = tuple(
        3.0 * tables.elements[first_z - 1].r4r2 * tables.elements[second_z - 1].r4r2
        for first_z, second_z in context.pair_elements
    )
    rd_values = tuple(spec.a1 * math.sqrt(value) + spec.a2 for value in rr_values)
    r6 = power(context.distance, 6)
    r8 = power(context.distance, 8)
    term6 = divide(
        _pair_constant(context, (spec.s6,) * pair_count),
        add(
            r6,
            _pair_constant(
                context,
                (value**6 for value in rd_values),
            ),
        ),
    )
    term8 = divide(
        _pair_constant(
            context,
            (spec.s8 * value for value in rr_values),
        ),
        add(
            r8,
            _pair_constant(
                context,
                (value**8 for value in rd_values),
            ),
        ),
    )
    damping = multiply(
        _switch(context, state, spec),
        add(term6, term8),
    )
    return multiply(
        _pair_constant(context, (-1.0,) * pair_count),
        multiply(c6, damping),
    )


@dataclass(frozen=True)
class D3GeometryProgram:
    """Compiler contract for one fixed D3 pair-state and its generated gradient."""

    spec: D3CompilerSpec
    geometry: GeometryIR
    pair_state: D3PairTopology
    program: Program
    version: str = D3_COMPILER_VERSION

    def __post_init__(self) -> None:
        if self.version != D3_COMPILER_VERSION:
            raise ValueError("unsupported D3 compiler lowering version")
        if self.geometry.parameter_identity != self.spec.identity:
            raise ValueError("D3 program GeometryIR/specification identity mismatch")
        if self.pair_state.spec_identity != self.spec.identity:
            raise ValueError("D3 program pair-state/specification identity mismatch")
        if self.pair_state.topology.atom_count != self.geometry.atom_count:
            raise ValueError("D3 program geometry/topology atom counts disagree")

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict:
        return {
            "kind": "d3-bj-geometry-pair-program",
            "version": self.version,
            "specification": self.spec.to_payload(),
            "geometry": self.geometry.to_payload(),
            "pair_state": self.pair_state.to_payload(),
            "equation": self.program.logical_hash,
        }

    def validate_execution_identity(self, identity: str) -> None:
        if identity != self.identity:
            raise ValueError("stale D3 compiler execution state")

    def validate_coordinates(self, coordinates: object) -> None:
        current = build_d3_pair_topology(
            self.geometry,
            self.spec,
            coordinates,
        )
        if current.identity != self.pair_state.identity:
            raise ValueError(
                "stale D3 pair topology/switch state for changed coordinates"
            )

    def coordinate_vjp(self) -> VJPProgram:
        return transpose_program(
            self.program,
            ["energy"],
            inputs=[self.geometry.coordinate_name],
        )


@dataclass(frozen=True)
class D3GeometryBatchProgram:
    """One generated D3(BJ) program for a heterogeneous ragged molecular batch."""

    spec: D3CompilerSpec
    geometry: GeometryIR
    system_atom_offsets: tuple[int, ...]
    pair_state: D3PairTopology
    program: Program
    version: str = D3_COMPILER_VERSION

    def __post_init__(self) -> None:
        if self.version != D3_COMPILER_VERSION:
            raise ValueError("unsupported D3 compiler lowering version")
        if self.geometry.parameter_identity != self.spec.identity:
            raise ValueError("D3 batch GeometryIR/specification identity mismatch")
        if self.pair_state.spec_identity != self.spec.identity:
            raise ValueError("D3 batch pair-state/specification identity mismatch")
        offsets = _normalized_d3_system_atom_offsets(
            self.system_atom_offsets, self.geometry.atom_count
        )
        object.__setattr__(self, "system_atom_offsets", offsets)
        if self.pair_state.topology.atom_count != self.geometry.atom_count:
            raise ValueError("D3 batch geometry/topology atom counts disagree")
        _d3_pair_system_owners(self.pair_state.topology, offsets)

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict:
        return {
            "kind": "d3-bj-geometry-ragged-batch-program",
            "version": self.version,
            "specification": self.spec.to_payload(),
            "geometry": self.geometry.to_payload(),
            "system_atom_offsets": list(self.system_atom_offsets),
            "pair_state": self.pair_state.to_payload(),
            "equation": self.program.logical_hash,
        }

    def validate_execution_identity(self, identity: str) -> None:
        if identity != self.identity:
            raise ValueError("stale D3 batch compiler execution state")

    def validate_coordinates(self, coordinates: object) -> None:
        current = build_d3_batch_pair_topology(
            self.geometry,
            self.spec,
            self.system_atom_offsets,
            coordinates,
        )
        if current.identity != self.pair_state.identity:
            raise ValueError(
                "stale D3 batch pair topology/switch state for changed coordinates"
            )

    def coordinate_vjp(self) -> VJPProgram:
        return transpose_program(
            self.program,
            ["energy"],
            inputs=[self.geometry.coordinate_name],
        )


def build_d3_geometry_program(
    spec: D3SpecLike | D3CompilerSpec,
    geometry: GeometryIR,
    pair_state: D3PairTopology,
) -> D3GeometryProgram:
    """Lower CN, C6 interpolation, BJ energy, and generated dE/dR to TensorIR."""

    compiler_spec = D3CompilerSpec.from_spec(spec)
    if geometry.parameter_identity != compiler_spec.identity:
        raise ValueError(
            "D3 GeometryIR parameter identity does not match specification"
        )
    if pair_state.spec_identity != compiler_spec.identity:
        raise ValueError("D3 pair state does not match specification")
    if pair_state.topology.atom_count != geometry.atom_count:
        raise ValueError("D3 geometry/topology atom counts disagree")

    context = lower_geometry(
        geometry,
        pair_state.topology,
    )
    coordination = _coordination(context, pair_state)
    weights = _reference_weights(geometry, coordination)
    c6 = _interpolated_c6(context, weights)
    pair_energy = _pair_energy(
        context,
        pair_state,
        compiler_spec,
        c6,
    )
    energy = pair_to_system(pair_energy, context)
    program = Program(
        {
            "energy": energy,
            "coordination": coordination,
            "c6": c6,
            "pair_energy": pair_energy,
        },
        provenance={
            "kind": "d3-bj-geometry-pair-ir",
            "version": D3_COMPILER_VERSION,
            "specification": compiler_spec.to_payload(),
            "table_sha256": D3_TABLE_SHA256,
            "radii_sha256": D3_RADII_SHA256,
            "pair_state": pair_state.to_payload(),
        },
    )
    return D3GeometryProgram(
        compiler_spec,
        geometry,
        pair_state,
        program,
    )


def compile_d3_bj(
    spec: D3SpecLike | D3CompilerSpec,
    elements: typing.Iterable[int],
    coordinates: object,
    *,
    coordinate_name: str = "coordinates",
) -> D3GeometryProgram:
    """Compile one molecular D3(BJ) pair state through GeometryIR/PairIR/TensorIR."""

    compiler_spec = D3CompilerSpec.from_spec(spec)
    geometry = d3_geometry(
        elements,
        compiler_spec,
        coordinate_name=coordinate_name,
    )
    pair_state = build_d3_pair_topology(
        geometry,
        compiler_spec,
        coordinates,
    )
    return build_d3_geometry_program(
        compiler_spec,
        geometry,
        pair_state,
    )


def build_d3_batch_geometry_program(
    spec: D3SpecLike | D3CompilerSpec,
    geometry: GeometryIR,
    system_atom_offsets: typing.Iterable[int],
    pair_state: D3PairTopology,
) -> D3GeometryBatchProgram:
    """Lower one heterogeneous ragged D3(BJ) batch through shared TensorIR."""

    compiler_spec = D3CompilerSpec.from_spec(spec)
    if geometry.parameter_identity != compiler_spec.identity:
        raise ValueError(
            "D3 GeometryIR parameter identity does not match specification"
        )
    offsets = _normalized_d3_system_atom_offsets(
        system_atom_offsets, geometry.atom_count
    )
    if pair_state.spec_identity != compiler_spec.identity:
        raise ValueError("D3 batch pair state does not match specification")
    if pair_state.topology.atom_count != geometry.atom_count:
        raise ValueError("D3 batch geometry/topology atom counts disagree")
    pair_system_owners = _d3_pair_system_owners(pair_state.topology, offsets)

    context = lower_geometry(
        geometry,
        pair_state.topology,
    )
    coordination = _coordination(context, pair_state)
    weights = _reference_weights(geometry, coordination)
    c6 = _interpolated_c6(context, weights)
    pair_energy = _pair_energy(
        context,
        pair_state,
        compiler_spec,
        c6,
    )
    system_index = Index(
        "s",
        IndexSpace("system", "batch", len(offsets) - 1),
    )
    energy = scatter_add(
        pair_energy,
        0,
        pair_system_owners,
        system_index,
    )
    program = Program(
        {
            "energy": energy,
            "coordination": coordination,
            "c6": c6,
            "pair_energy": pair_energy,
        },
        provenance={
            "kind": "d3-bj-geometry-pair-ir-ragged-batch",
            "version": D3_COMPILER_VERSION,
            "specification": compiler_spec.to_payload(),
            "table_sha256": D3_TABLE_SHA256,
            "radii_sha256": D3_RADII_SHA256,
            "system_atom_offsets": list(offsets),
            "pair_state": pair_state.to_payload(),
        },
    )
    return D3GeometryBatchProgram(
        compiler_spec,
        geometry,
        offsets,
        pair_state,
        program,
    )


def compile_d3_bj_batch(
    spec: D3SpecLike | D3CompilerSpec,
    elements: typing.Iterable[int],
    system_atom_offsets: typing.Iterable[int],
    coordinates: object,
    *,
    coordinate_name: str = "coordinates",
) -> D3GeometryBatchProgram:
    """Compile one heterogeneous ragged D3(BJ) batch through PairIR/TensorIR."""

    compiler_spec = D3CompilerSpec.from_spec(spec)
    geometry = d3_geometry(
        elements,
        compiler_spec,
        coordinate_name=coordinate_name,
    )
    offsets = _normalized_d3_system_atom_offsets(
        system_atom_offsets, geometry.atom_count
    )
    pair_state = build_d3_batch_pair_topology(
        geometry,
        compiler_spec,
        offsets,
        coordinates,
    )
    return build_d3_batch_geometry_program(
        compiler_spec,
        geometry,
        offsets,
        pair_state,
    )
