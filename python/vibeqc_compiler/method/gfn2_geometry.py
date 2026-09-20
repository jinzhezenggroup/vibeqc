"""Method-layer binding for the GFN2 short-range geometry compiler (#504)."""

from __future__ import annotations

from dataclasses import dataclass

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.geometry.gfn2 import (
    GFN2_SHORT_RANGE_PARAMETER_IDENTITY,
    GFN2_SHORT_RANGE_VERSION,
    GeometryIR,
    PairTopology,
    build_gfn2_pair_topology,
    build_gfn2_short_range_program,
)
from vibeqc_compiler.tensor import Program, transpose_program

from .xtb import GFN2_PARAMETER_SET, XtbMethodIR, resolve_xtb_method


@dataclass(frozen=True)
class Gfn2GeometryProgram:
    method: XtbMethodIR
    geometry: GeometryIR
    topology: PairTopology
    program: Program
    parameter_identity: str = GFN2_SHORT_RANGE_PARAMETER_IDENTITY
    version: str = GFN2_SHORT_RANGE_VERSION

    def __post_init__(self) -> None:
        if self.method.model_flavor != "gfn2":
            raise ValueError("GFN2 geometry program requires the GFN2 method graph")
        if self.method.parameter_set.identity != GFN2_PARAMETER_SET.identity:
            raise ValueError(
                "GFN2 geometry program requires the audited parameter manifest"
            )
        if self.geometry.parameter_identity != self.parameter_identity:
            raise ValueError("GFN2 geometry parameter identity mismatch")
        if self.version != GFN2_SHORT_RANGE_VERSION:
            raise ValueError("unsupported GFN2 short-range compiler version")

    @property
    def identity(self) -> str:
        return canonical_hash(
            {
                "version": self.version,
                "method": self.method.identity,
                "geometry": self.geometry.to_payload(),
                "topology": self.topology.to_payload(),
                "equation": self.program.logical_hash,
                "parameter_identity": self.parameter_identity,
            }
        )

    def validate_execution_identity(self, identity: str) -> None:
        if identity != self.identity:
            raise ValueError("stale GFN2 geometry compiler execution state")

    def validate_coordinates(self, coordinates) -> None:
        expected = build_gfn2_pair_topology(self.geometry, coordinates)
        if expected.identity != self.topology.identity:
            raise ValueError("stale GFN2 pair topology for changed coordinates")

    def coordinate_vjp(self, output: str):
        if "nuclear-gradient" not in self.method.requested_products:
            raise ValueError(
                "GFN2 coordinate VJP requires nuclear-gradient compiler product"
            )
        if output not in ("coordination", "repulsion_energy"):
            raise ValueError("unknown GFN2 short-range derivative output")
        return transpose_program(
            self.program,
            [output],
            inputs=[self.geometry.coordinate_name],
        )


def build_gfn2_geometry_program(
    method: str | XtbMethodIR,
    geometry: GeometryIR,
    topology: PairTopology,
) -> Gfn2GeometryProgram:
    """Bind the geometry-only GFN2 TensorIR graph to an audited method graph."""

    if isinstance(method, str):
        method = resolve_xtb_method(
            method,
            requested_products=("energy", "nuclear-gradient"),
        )
    elif not isinstance(method, XtbMethodIR):
        raise TypeError("method must be a GFN2 catalog name or XtbMethodIR")

    if method.model_flavor != "gfn2":
        raise ValueError("GFN2 short-range lowering requires model_flavor='gfn2'")
    if method.parameter_set.identity != GFN2_PARAMETER_SET.identity:
        raise ValueError(
            "GFN2 short-range lowering requires the audited parameter manifest"
        )

    program = build_gfn2_short_range_program(geometry, topology)
    return Gfn2GeometryProgram(
        method,
        geometry,
        topology,
        program,
    )
