"""Development TensorIR: typed equations, CPU reference execution, and replay.

Integral operators remain in vibeqc_codegen. This package supplies no CCSD
method, CUDA lowering, or automatic differentiation implementation.
"""

from .interpreter import Execution, execute
from .ir import (
    PRIMITIVES,
    Node,
    add,
    broadcast,
    constant,
    divide,
    einsum,
    gather,
    input_tensor,
    multiply,
    reduce_sum,
    reshape,
    slice_tensor,
    transpose,
)
from .optimize import PASSES, optimize, rewrite
from .packing import PackedLayout
from .program import Program
from .types import Index, IndexSpace, Symmetry, TensorSpec

__all__ = [
    "PASSES",
    "PRIMITIVES",
    "Execution",
    "Index",
    "IndexSpace",
    "Node",
    "PackedLayout",
    "Program",
    "Symmetry",
    "TensorSpec",
    "add",
    "broadcast",
    "constant",
    "divide",
    "einsum",
    "execute",
    "gather",
    "input_tensor",
    "multiply",
    "optimize",
    "reduce_sum",
    "reshape",
    "rewrite",
    "slice_tensor",
    "transpose",
]
