"""Development TensorIR: typed equations, CPU execution, replay, and AD.

Integral operators remain in vibeqc_compiler.integral.  This package supplies primitive
JVP/VJP rules, demand-driven derivative programs, packed-layout adjoints, and
plans for the existing CUDA lowering path; it supplies no CCSD method or
complete solver.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from vibeqc_compiler.common.layout import DenseLayout

from .ad_program import (
    GENERATION_VERSION,
    JVPProgram,
    VJPProgram,
    linearize,
    transpose_program,
)
from .batch_schedule import (
    BATCH_SCHEDULE_SCHEMA,
    BatchScheduleIR,
    RaggedStepSchedule,
    analyze_batch_schedule,
)
from .ir import (
    PRIMITIVES,
    Node,
    add,
    broadcast,
    cast,
    constant,
    divide,
    einsum,
    exp,
    gather,
    indexed_gather,
    input_tensor,
    log,
    multiply,
    power,
    reduce_sum,
    reshape,
    runtime_indexed_scatter_add,
    runtime_indexed_select,
    scaled_bilinear,
    scatter_add,
    segment_sum,
    slice_tensor,
    sqrt,
    transpose,
)
from .optimize import PASSES, optimize, rewrite
from .precision import (
    CastBoundary,
    PrecisionDirective,
    PrecisionSchedule,
    ValuePrecision,
    conservative_precision_variants,
    describe_precision,
    lower_precision,
)
from .program import Program
from .scf import (
    SCF_TENSOR_VERSION,
    density_program,
    diis_extrapolation_program,
    diis_gram_program,
    energy_program,
    fock_composition_program,
    weighted_density_program,
)
from .types import Index, IndexSpace, Symmetry, TensorSpec

if TYPE_CHECKING:
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
    from .packing import PackedLayout


# Keep compiler/codegen-only imports dependency-light.  NumPy-backed reference
# execution, AD, and packing remain available through the public
# package API but load only when a caller actually asks for them.
_LAZY_EXPORTS = {
    "AD_PRIMITIVES": ("autodiff", "AD_PRIMITIVES"),
    "AD_RULE_VERSION": ("autodiff", "AD_RULE_VERSION"),
    "AD_RULES": ("autodiff", "AD_RULES"),
    "DotTestResult": ("autodiff", "DotTestResult"),
    "JVPResult": ("autodiff", "JVPResult"),
    "VJPResult": ("autodiff", "VJPResult"),
    "capabilities": ("autodiff", "capabilities"),
    "dot_test": ("autodiff", "dot_test"),
    "jvp": ("autodiff", "jvp"),
    "vjp": ("autodiff", "vjp"),
    "Execution": ("interpreter", "Execution"),
    "execute": ("interpreter", "execute"),
    "PackedLayout": ("packing", "PackedLayout"),
}


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "AD_PRIMITIVES",
    "AD_RULES",
    "AD_RULE_VERSION",
    "BATCH_SCHEDULE_SCHEMA",
    "GENERATION_VERSION",
    "PASSES",
    "PRIMITIVES",
    "SCF_TENSOR_VERSION",
    "BatchScheduleIR",
    "CastBoundary",
    "DenseLayout",
    "DotTestResult",
    "Execution",
    "Index",
    "IndexSpace",
    "JVPProgram",
    "JVPResult",
    "Node",
    "PackedLayout",
    "PrecisionDirective",
    "PrecisionSchedule",
    "Program",
    "RaggedStepSchedule",
    "Symmetry",
    "TensorSpec",
    "VJPProgram",
    "VJPResult",
    "ValuePrecision",
    "add",
    "analyze_batch_schedule",
    "broadcast",
    "capabilities",
    "cast",
    "conservative_precision_variants",
    "constant",
    "density_program",
    "describe_precision",
    "diis_extrapolation_program",
    "diis_gram_program",
    "divide",
    "dot_test",
    "einsum",
    "energy_program",
    "execute",
    "exp",
    "fock_composition_program",
    "gather",
    "indexed_gather",
    "input_tensor",
    "jvp",
    "linearize",
    "log",
    "lower_precision",
    "multiply",
    "optimize",
    "power",
    "reduce_sum",
    "reshape",
    "rewrite",
    "runtime_indexed_scatter_add",
    "runtime_indexed_select",
    "scaled_bilinear",
    "scatter_add",
    "segment_sum",
    "slice_tensor",
    "sqrt",
    "transpose",
    "transpose_program",
    "vjp",
    "weighted_density_program",
]
