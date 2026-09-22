"""Compose generated GFN1 halogen TripletIR with canonical XtbMethodIR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.geometry.gfn1 import (
    GFN1_HALOGEN_PARAMETER_IDENTITY,
    Gfn1HalogenGeometryProgram,
    build_gfn1_halogen_geometry_program,
    build_gfn1_halogen_topology,
    gfn1_geometry,
)

from .xtb import GFN1_PARAMETER_SET, XtbMethodIR, XtbPrimitive, resolve_xtb_method

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import ArrayLike

    from vibeqc_compiler.geometry.ir import GeometryIR
    from vibeqc_compiler.geometry.triplet import TripletTopology
    from vibeqc_compiler.tensor import Program


GFN1_HALOGEN_METHOD_VERSION = "gfn1-halogen-method-ir-v1"


def _halogen_primitive(method: XtbMethodIR) -> XtbPrimitive:
    if method.model_flavor != "gfn1" or method.parameter_set != GFN1_PARAMETER_SET:
        raise ValueError(
            "GFN1 halogen lowering requires the canonical GFN1 XtbMethodIR"
        )
    primitive = next(
        (
            primitive
            for primitive in method.primitives
            if primitive.kind == "halogen_correction"
        ),
        None,
    )
    if primitive is None:
        raise ValueError("GFN1 XtbMethodIR is missing the halogen correction primitive")
    if (
        primitive.model != "gfn1-halogen-correction"
        or primitive.parameter_domains != ("correction",)
        or primitive.derivative_capabilities != ("energy", "nuclear-gradient")
        or primitive.self_consistent
    ):
        raise ValueError("GFN1 halogen primitive semantics are not canonical")
    return primitive


@dataclass(frozen=True)
class Gfn1HalogenProgram:
    """Method-bound GFN1 halogen energy with a generated Cartesian VJP."""

    method: XtbMethodIR
    geometry_program: Gfn1HalogenGeometryProgram
    primitive_identity: str
    version: str = GFN1_HALOGEN_METHOD_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.method, XtbMethodIR):
            raise TypeError("GFN1 halogen composition requires XtbMethodIR")
        primitive = _halogen_primitive(self.method)
        expected = canonical_hash(primitive.semantic_payload())
        if self.primitive_identity != expected:
            raise ValueError("GFN1 halogen primitive identity mismatch")
        if not isinstance(self.geometry_program, Gfn1HalogenGeometryProgram):
            raise TypeError("GFN1 halogen composition requires a geometry program")
        if self.version != GFN1_HALOGEN_METHOD_VERSION:
            raise ValueError("unsupported GFN1 halogen MethodIR composition version")

    @property
    def geometry(self) -> GeometryIR:
        return self.geometry_program.geometry

    @property
    def topology(self) -> TripletTopology:
        return self.geometry_program.topology

    @property
    def program(self) -> Program:
        return self.geometry_program.program

    @property
    def parameter_identity(self) -> str:
        return GFN1_HALOGEN_PARAMETER_IDENTITY

    @property
    def identity(self) -> str:
        return canonical_hash(
            {
                "version": self.version,
                "method_identity": self.method.identity,
                "primitive_identity": self.primitive_identity,
                "geometry_program_identity": self.geometry_program.identity,
            }
        )

    def validate_execution_identity(self, identity: str) -> None:
        if identity != self.identity:
            raise ValueError("stale GFN1 halogen MethodIR execution state")

    def validate_coordinates(self, coordinates: ArrayLike) -> None:
        self.geometry_program.validate_coordinates(coordinates)

    def _require_gradient_product(self) -> None:
        if "nuclear-gradient" not in self.method.requested_products:
            raise ValueError(
                "GFN1 halogen coordinate AD requires the nuclear-gradient compiler product"
            )

    def coordinate_jvp(self) -> object:
        self._require_gradient_product()
        return self.geometry_program.coordinate_jvp()

    def coordinate_vjp(self) -> object:
        self._require_gradient_product()
        return self.geometry_program.coordinate_vjp()


def build_gfn1_halogen_program(
    method: str | XtbMethodIR,
    geometry: GeometryIR,
    topology: TripletTopology,
) -> Gfn1HalogenProgram:
    """Bind the canonical MethodIR primitive to one fixed triplet topology."""

    if isinstance(method, str):
        method = resolve_xtb_method(
            method, requested_products=("energy", "nuclear-gradient")
        )
    elif not isinstance(method, XtbMethodIR):
        raise TypeError(
            "GFN1 halogen composition requires a method name or XtbMethodIR"
        )
    primitive = _halogen_primitive(method)
    geometry_program = build_gfn1_halogen_geometry_program(geometry, topology)
    return Gfn1HalogenProgram(
        method,
        geometry_program,
        canonical_hash(primitive.semantic_payload()),
    )


def compile_gfn1_halogen(
    method: str | XtbMethodIR,
    elements: Iterable[int],
    coordinates: ArrayLike,
    *,
    coordinate_name: str = "coordinates",
) -> Gfn1HalogenProgram:
    """Build topology and compile the canonical GFN1 halogen correction."""

    geometry = gfn1_geometry(elements, coordinate_name=coordinate_name)
    topology = build_gfn1_halogen_topology(geometry, coordinates)
    return build_gfn1_halogen_program(method, geometry, topology)
