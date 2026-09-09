"""Semantic lowering gates require no OpenCL SDK or accelerator."""

from dataclasses import replace

import pytest

from tools.vibeqc_codegen.cache import integral_cache_key
from tools.vibeqc_codegen.cuda import CudaEmitter
from tools.vibeqc_codegen.df_values import build_df_component_kernel, build_df_value_ir
from tools.vibeqc_codegen.opencl_lowering import ScalarKernel, emit_opencl, source_hash
from tools.vibeqc_codegen.runtime_backend import ExecutionShape, RuntimeCapabilities
from tools.vibeqc_codegen.scalar_c import ScalarCEmitter
from tools.vibeqc_validation.schema import canonical_hash


def integral_program():
    """Use the existing (p_x|p_y) auxiliary Coulomb expression unchanged."""
    integral = build_df_value_ir("coulomb_metric", (1, 1))
    kernel = build_df_component_kernel(integral, ("x", "y"))
    inputs = tuple(
        sorted(
            {
                str(kernel.graph.nodes[i].payload)
                for i in kernel.graph.topological_order((kernel.value,))
                if kernel.graph.nodes[i].operation == "variable"
            }
        )
    )
    return ScalarKernel(
        kernel.graph,
        (kernel.value,),
        inputs,
        canonical_hash(
            {"integral": integral_cache_key(integral), "components": kernel.components}
        ),
    )


def test_real_integral_dag_retains_identical_cuda_scalar_arithmetic():
    kernel = integral_program()
    emitters = [emitter(kernel.graph, {}) for emitter in (CudaEmitter, ScalarCEmitter)]
    for emitter in emitters:
        emitter.emit(kernel.roots)
    assert emitters[0].lines == emitters[1].lines
    assert emitters[0].reference(kernel.roots[0]) == emitters[1].reference(
        kernel.roots[0]
    )
    target = RuntimeCapabilities("opencl", True, 256, 32768)
    first = emit_opencl(kernel, target, ExecutionShape(16))
    second = emit_opencl(kernel, target, ExecutionShape(64))
    assert kernel.scientific_hash in first and kernel.scientific_hash in second
    assert source_hash(first) != source_hash(second)
    assert "get_global_id(0)" in first and "__global const double* inputs" in first
    assert "__device__" not in first and "threadIdx" not in first
    assert "if (item >= count) return" in first


def test_missing_scientific_inputs_or_injected_identifiers_are_rejected():
    kernel = integral_program()
    with pytest.raises(ValueError, match="no primitive ABI column"):
        replace(kernel, inputs=kernel.inputs[:-1])
    for name in (
        "x); injected()",
        "inputs",
        "v0",
        "double",
        "return",
        "_Atomic",
        "sqrt",
    ):
        with pytest.raises(ValueError, match="identifier"):
            replace(kernel, name=name)
