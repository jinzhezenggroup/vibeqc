"""Curated public subset of the versioned TensorIR construction API.

CUDA schedules, emitters, compiler processes and artifact loaders intentionally
remain compiler APIs. This module exposes only mathematical IR construction,
serialization, optimization, AD and interpreter operations.
"""

from __future__ import annotations

from vibeqc_compiler.tensor import (
    DenseLayout,
    Index,
    IndexSpace,
    Node,
    PackedLayout,
    Program,
    Symmetry,
    TensorSpec,
    add,
    broadcast,
    capabilities,
    constant,
    divide,
    dot_test,
    einsum,
    execute,
    exp,
    gather,
    input_tensor,
    jvp,
    linearize,
    log,
    multiply,
    optimize,
    power,
    reduce_sum,
    reshape,
    scaled_bilinear,
    slice_tensor,
    sqrt,
    transpose,
    transpose_program,
    vjp,
)
from vibeqc_compiler.tensor.program import PRIMITIVE_VERSION
from vibeqc_compiler.tensor.program import VERSION as SCHEMA_VERSION

API_VERSION = 1


def inspect(program: Program) -> dict:
    """Return a detached, replayable TensorIR description."""
    if not isinstance(program, Program):
        raise TypeError("tensor inspection requires Program")
    return {
        "extension_api_version": API_VERSION,
        "kind": "tensor",
        "logical_hash": program.logical_hash,
        "program": program.to_payload(),
    }


__all__ = [
    "API_VERSION",
    "PRIMITIVE_VERSION",
    "SCHEMA_VERSION",
    "DenseLayout",
    "Index",
    "IndexSpace",
    "Node",
    "PackedLayout",
    "Program",
    "Symmetry",
    "TensorSpec",
    "add",
    "broadcast",
    "capabilities",
    "constant",
    "divide",
    "dot_test",
    "einsum",
    "execute",
    "exp",
    "gather",
    "input_tensor",
    "inspect",
    "jvp",
    "linearize",
    "log",
    "multiply",
    "optimize",
    "power",
    "reduce_sum",
    "reshape",
    "scaled_bilinear",
    "slice_tensor",
    "sqrt",
    "transpose",
    "transpose_program",
    "vjp",
]
