"""Development TensorIR: typed equations, CPU execution, replay, and AD.

Integral operators remain in vibeqc_compiler.integral.  This package supplies primitive
JVP/VJP rules, demand-driven derivative programs, packed-layout adjoints, and
plans for the existing CUDA lowering path; it supplies no CCSD method or
complete solver.
"""

from .ad_program import (
    GENERATION_VERSION,
    JVPProgram,
    VJPProgram,
    linearize,
    transpose_program,
)
from .autodiff import (
    AD_PRIMITIVES,
    AD_RULE_VERSION,
    AD_RULES,
    DotTestResult,
    JVPResult,
    VJPResult,
    capabilities,
    dot_test,
    jvp,
    vjp,
)
from .interpreter import Execution, execute
from .ir import (
    PRIMITIVES,
    Node,
    add,
    broadcast,
    constant,
    divide,
    einsum,
    exp,
    gather,
    input_tensor,
    log,
    multiply,
    power,
    reduce_sum,
    reshape,
    scaled_bilinear,
    slice_tensor,
    sqrt,
    transpose,
)
from .layout import DenseLayout
from .optimize import PASSES, optimize, rewrite
from .packing import PackedLayout
from .program import Program
from .types import Index, IndexSpace, Symmetry, TensorSpec

__all__ = [
    "AD_PRIMITIVES",
    "AD_RULES",
    "AD_RULE_VERSION",
    "GENERATION_VERSION",
    "PASSES",
    "PRIMITIVES",
    "DenseLayout",
    "DotTestResult",
    "Execution",
    "Index",
    "IndexSpace",
    "JVPProgram",
    "JVPResult",
    "Node",
    "PackedLayout",
    "Program",
    "Symmetry",
    "TensorSpec",
    "VJPProgram",
    "VJPResult",
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
    "jvp",
    "linearize",
    "log",
    "multiply",
    "optimize",
    "power",
    "reduce_sum",
    "reshape",
    "rewrite",
    "scaled_bilinear",
    "slice_tensor",
    "sqrt",
    "transpose",
    "transpose_program",
    "vjp",
]
