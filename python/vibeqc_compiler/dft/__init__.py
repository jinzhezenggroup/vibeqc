"""Compiler-side DFT grid/AO interfaces and canonical method composition IR."""

from .ao import NativeAO, jet_indices
from .density_source import DensitySource, DensityStamp
from .features import density_features, orbital_features, spin_densities
from .grid import ExplicitGrid, GridSpec, MolecularGrid, partition_weights
from .method import (
    METHOD_CATALOG,
    ExactExchangePrimitive,
    MethodIR,
    MethodSpec,
    SemilocalXCPrimitive,
    UnsupportedMethod,
    resolve_method,
)
from .prepared import PreparedGrid, PreparedGridBatch

__all__ = [
    "METHOD_CATALOG",
    "DensitySource",
    "DensityStamp",
    "ExactExchangePrimitive",
    "ExplicitGrid",
    "GridSpec",
    "MethodIR",
    "MethodSpec",
    "MolecularGrid",
    "NativeAO",
    "PreparedGrid",
    "PreparedGridBatch",
    "SemilocalXCPrimitive",
    "UnsupportedMethod",
    "density_features",
    "jet_indices",
    "orbital_features",
    "partition_weights",
    "resolve_method",
    "spin_densities",
]
