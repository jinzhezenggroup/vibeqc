"""Geometry/pair compiler contracts lowered through the shared TensorIR."""

from .ir import (
    LOWERING_VERSION,
    PAIR_OWNERSHIP,
    GeometryIR,
    PairCutoff,
    PairProgram,
    PairTensorContext,
    PairTopology,
    build_pair_program,
    inverse_power_program,
    lower_geometry,
    pair_to_atom,
    pair_to_system,
)

__all__ = [
    "LOWERING_VERSION",
    "PAIR_OWNERSHIP",
    "GeometryIR",
    "PairCutoff",
    "PairProgram",
    "PairTensorContext",
    "PairTopology",
    "build_pair_program",
    "inverse_power_program",
    "lower_geometry",
    "pair_to_atom",
    "pair_to_system",
]
