"""Emit one native CUDA executor per immutable TensorIR execution plan.

Python generates the whole launch/contraction sequence ahead of time. Runtime
host code only moves data and schedules device work; all tensor arithmetic,
including general einsums and denominators, executes on the allocated GPU.
"""

from __future__ import annotations

import typing
from math import prod

from .cuda_dtype import scalar_type
from .cuda_gemm import gemm_contract
from .cuda_plan import (
    ALIGNMENT,
    TensorPlan,
    _index_table_values,
    aligned,
    static_data_slices,
    strides,
)
from .cuda_reduction import cooperative_reduction_provider
from .ir import TRANSCENDENTALS
from .scaled_arithmetic import emit_scaled_bilinear


def _integer(value: typing.Any) -> typing.Any:
    return f"{value}LL"


def _coordinate(linear: typing.Any, shape: typing.Any, axis: typing.Any) -> typing.Any:
    # Empty kernels/accessors are never executed, but CUDA still compiles
    # their bodies. Avoid constant division by zero even in unreachable code.
    return f"(({linear}) / {_integer(max(1, prod(shape[axis + 1 :])))} % {_integer(max(1, shape[axis]))})"


def _flat(coordinates: typing.Any, shape: typing.Any) -> typing.Any:
    return (
        " + ".join(
            f"({coord}) * {_integer(stride)}"
            for coord, stride in zip(coordinates, strides(shape), strict=True)
        )
        or "0LL"
    )


def _physical_index(layout: typing.Any, logical: typing.Any = "z") -> typing.Any:
    if layout.is_c_contiguous:
        return logical
    return (
        " + ".join(
            f"({_coordinate(logical, layout.shape, axis)}) * {_integer(stride)}"
            for axis, stride in enumerate(layout.element_strides)
        )
        or "0LL"
    )


def _logical_index(layout: typing.Any, physical: typing.Any = "z") -> typing.Any:
    if layout.is_c_contiguous:
        return physical
    shape = tuple(layout.shape[axis] for axis in layout.order)
    coordinates = [None] * len(shape)
    for axis, logical_axis in enumerate(layout.order):
        coordinates[logical_axis] = _coordinate(physical, shape, axis)
    return _flat(coordinates, layout.shape)


def _name(prefix: typing.Any, base: typing.Any) -> typing.Any:
    return f"{prefix}{base}"


def _read(
    operand: typing.Any, index: typing.Any, prefix: typing.Any = ""
) -> typing.Any:
    return f"{_name(prefix, f'read_{operand}')}(p, {index}, error)"


def _value_precision(plan: typing.Any, i: int) -> typing.Any:
    return plan.precision_by_node[plan.steps[i].node]


def _reduce_source_index(
    node: typing.Any, logical: str = "z", reduction: str = "r"
) -> tuple[str, int]:
    """Return the flattened source index and reduction extent for reduce."""

    source_shape = node.inputs[0].spec.shape
    reduction_shape = tuple(source_shape[axis] for axis in node.attrs["axes"])
    output_coordinates = [
        _coordinate(logical, node.spec.shape, axis)
        for axis in range(len(node.spec.shape))
    ]
    source, cursor = [], 0
    for axis in range(len(source_shape)):
        if axis in node.attrs["axes"]:
            source.append(
                _coordinate(reduction, reduction_shape, node.attrs["axes"].index(axis))
            )
        else:
            source.append(output_coordinates[cursor])
            cursor += 1
    return _flat(source, source_shape), prod(reduction_shape)


def _convert(value: str, source: typing.Any, target: typing.Any) -> str:
    if source.dtype == target.dtype:
        return value
    if source.dtype == "float64" and target.dtype == "float32":
        return f"__double2float_rn({value})"
    if source.dtype == "float32" and target.dtype == "float64":
        return f"static_cast<double>({value})"
    raise ValueError("unsupported TensorIR CUDA precision conversion")


def _value(plan: typing.Any, i: typing.Any, prefix: typing.Any = "") -> typing.Any:
    """Emit scalar evaluation with each original arithmetic error boundary."""
    step = plan.steps[i]
    node, a, args = step.node, step.node.attrs, step.inputs
    if node.spec.dtype == "int64":
        raise ValueError("int64 TensorIR controls are read-only CUDA inputs")
    scalar = scalar_type(node.spec.dtype)
    precision = _value_precision(plan, i)
    accumulator = scalar_type(precision.accumulation_dtype)
    ty, add, mul = scalar.ctype, scalar.intrinsic("add"), scalar.intrinsic("mul")
    acc_ty = accumulator.ctype
    acc_add = accumulator.intrinsic("add")
    shape = node.spec.shape
    c = [_coordinate("z", shape, axis) for axis in range(len(shape))]
    reduction_pragma = (
        ""
        if plan.schedule.reduction_unroll == 1
        else f"#pragma unroll {plan.schedule.reduction_unroll}\n"
    )
    if node.op == "cast":
        child = args[0]
        source = scalar_type(plan.steps[child].node.spec.dtype)
        value = _read(child, "z", prefix)
        # FP32 -> FP64 is exact; FP64 -> FP32 is explicit RN conversion.
        converted = _convert(value, source, scalar)
        return f"return finite({converted}, error, {i});"
    if node.op == "add":
        lines = [f"{ty} value = {scalar.zero};"]
        for child, factor in zip(args, a["coefficients"], strict=True):
            lines.append(
                f"value = {add}(value, {mul}({scalar.literal(factor)}, {_read(child, 'z', prefix)}));"
            )
        return "\n".join(lines + [f"return finite(value, error, {i});"])
    if node.op == "multiply":
        return f"return finite({mul}({_read(args[0], 'z', prefix)}, {_read(args[1], 'z', prefix)}), error, {i});"
    if node.op == "divide":
        return f"return quotient({_read(args[0], 'z', prefix)}, {_read(args[1], 'z', prefix)}, error, {i});"
    if node.op == "scaled_bilinear":
        operands = ", ".join(_read(child, "z", prefix) for child in args)
        return f"return {prefix}scaled_bilinear({operands}, error, {i});"
    if node.op in TRANSCENDENTALS:
        lines = [f"const {ty} x = {_read(args[0], 'z', prefix)};"]
        if node.op in ("log", "power", "sqrt"):
            predicate = "x >= 0.0" if node.op == "sqrt" else "x > 0.0"
            lines.append(
                f"if (!({predicate})) {{ atomicCAS(error, 0, -{len(plan.steps) + i + 1}); return 0.0; }}"
            )
        function = ("pow" if node.op == "power" else node.op) + scalar.suffix
        arguments = "x"
        if node.op == "power":
            arguments += ", " + scalar.literal(a["exponent"])
        lines.append(f"return finite(::{function}({arguments}), error, {i});")
        return "\n".join(lines)
    if node.op == "einsum":
        domains = {}
        for child, labels in zip(node.inputs, a["labels"], strict=True):
            domains.update(zip(labels, child.spec.shape, strict=True))
        reduced = tuple(label for label in sorted(domains) if label not in a["output"])
        reduction_shape = tuple(domains[label] for label in reduced)
        mapping = dict(zip(a["output"], c, strict=True))
        mapping.update(
            (label, _coordinate("r", reduction_shape, axis))
            for axis, label in enumerate(reduced)
        )
        values = [
            _read(
                child,
                _flat(
                    [mapping[label] for label in labels],
                    plan.steps[child].node.spec.shape,
                ),
                prefix,
            )
            for child, labels in zip(args, a["labels"], strict=True)
        ]
        term = values[0]
        for value in values[1:]:
            term = f"{mul}({term}, {value})"
        accumulated_term = _convert(term, scalar, accumulator)
        narrowed = _convert(f"finite(value, error, {i})", accumulator, scalar)
        scaled = f"{mul}({narrowed}, {scalar.literal(a['coefficient'])})"
        return f"""{acc_ty} value = {accumulator.zero};
{reduction_pragma}for (I r = 0; r < {_integer(prod(reduction_shape))}; ++r)
    value = {acc_add}(value, {accumulated_term});
return finite({scaled}, error, {i});"""
    if node.op == "runtime_indexed_select":
        child, maps = args[0], args[1:]
        source_shape = plan.steps[child].node.spec.shape
        domain = c[0]
        selected = {}
        lines = []
        error_code = -(2 * len(plan.steps) + i + 1)
        for ordinal, (axis, mapping) in enumerate(zip(a["axes"], maps, strict=True)):
            variable = f"runtime_index_{ordinal}"
            lines.append(f"const I {variable} = {_read(mapping, domain, prefix)};")
            lines.append(
                f"if ({variable} < 0 || {variable} >= {_integer(source_shape[axis])}) "
                f"{{ atomicCAS(error, 0, {error_code}); return {scalar.zero}; }}"
            )
            selected[axis] = variable
        coordinates = []
        output_axis = 1
        for axis in range(len(source_shape)):
            if axis in selected:
                coordinates.append(selected[axis])
            else:
                coordinates.append(c[output_axis])
                output_axis += 1
        lines.append(
            f"return {_read(child, _flat(coordinates, source_shape), prefix)};"
        )
        return "\n".join(lines)
    child = args[0]
    source_shape = plan.steps[child].node.spec.shape
    if node.op == "reshape":
        index = "z"
    elif node.op == "transpose":
        source = [c[a["axes"].index(axis)] for axis in range(len(source_shape))]
        index = _flat(source, source_shape)
    elif node.op == "slice":
        index = _flat(
            [
                f"({coord} + {_integer(start)})"
                for coord, (start, _) in zip(c, a["ranges"], strict=True)
            ],
            source_shape,
        )
    elif node.op == "broadcast":
        index = _flat([c[axis] for axis in a["axes"]], source_shape)
    elif node.op in ("gather", "indexed_gather"):
        table = dict(plan.index_tables)[i]
        c[a["axis"]] = f"reinterpret_cast<const I*>(p + {table})[{c[a['axis']]}]"
        index = _flat(c, source_shape)
    elif node.op == "scatter_add":
        axis = a["axis"]
        if not source_shape[axis]:
            return f"return finite({scalar.zero}, error, {i});"
        table = dict(plan.index_tables)[i]
        target = c[axis]
        target_extent = shape[axis]
        source = list(c)
        source[axis] = "r"
        return (
            f"{ty} value = {scalar.zero};\n"
            f"const I* index = reinterpret_cast<const I*>(p + {table});\n"
            f"const I begin = index[{target}], end = index[{target} + 1];\n"
            f"{reduction_pragma}for (I q = begin; q < end; ++q) {{\n"
            f"    const I r = index[{target_extent + 1}LL + q];\n"
            f"    value = {add}(value, {_read(child, _flat(source, source_shape), prefix)});\n"
            "}\n"
            f"return finite(value, error, {i});"
        )
    elif node.op == "segment_sum":
        table = dict(plan.index_tables)[i]
        axis = a["axis"]
        segment = c[axis]
        source = list(c)
        source[axis] = "r"
        return f"""{ty} value = {scalar.zero};
const I begin = reinterpret_cast<const I*>(p + {table})[{segment}];
const I end = reinterpret_cast<const I*>(p + {table})[{segment} + 1];
{reduction_pragma}for (I r = begin; r < end; ++r)
    value = {add}(value, {_read(child, _flat(source, source_shape), prefix)});
return finite(value, error, {i});"""
    elif node.op == "reduce":
        source_index, reduction_size = _reduce_source_index(node)
        contribution = _convert(_read(child, source_index, prefix), scalar, accumulator)
        result = _convert("value", accumulator, scalar)
        return f"""{acc_ty} value = {accumulator.zero};
{reduction_pragma}for (I r = 0; r < {_integer(reduction_size)}; ++r)
    value = {acc_add}(value, {contribution});
return finite({result}, error, {i});"""
    else:
        raise ValueError(f"unsupported CUDA primitive: {node.op}")
    return f"return {_read(child, index, prefix)};"


def _arithmetic_error_expression(plan: typing.Any, legacy: typing.Any) -> typing.Any:
    """Extend diagnostics only for new graphs; keep legacy emitted bytes intact.

    Zero is success, +[1,n] is nonfinite, -[1,n] is division by zero,
    -[n+1,2n] is a scalar-domain error, and runtime index failures occupy a
    separate range below -2n. The planner bounds the integer diagnostic range.
    """
    transcendental = any(s.node.op in TRANSCENDENTALS for s in plan.steps)
    runtime_indexed = any(s.node.op == "runtime_indexed_select" for s in plan.steps)
    if not transcendental and not runtime_indexed:
        return legacy
    n = len(plan.steps)
    expression = legacy
    if transcendental:
        expression = (
            f"(arithmetic_error < -{n} ? "
            'std::string("tensor transcendental domain error at step ") + '
            f"std::to_string(-arithmetic_error - {n} - 1) : ({legacy}))"
        )
    if runtime_indexed:
        expression = (
            f"(arithmetic_error < -{2 * n} ? "
            'std::string("tensor runtime index out of bounds at step ") + '
            f"std::to_string(-arithmetic_error - {2 * n} - 1) : ({expression}))"
        )
    return expression


def _group_map(g: typing.Any, labels: typing.Any) -> typing.Any:
    coordinates = {}
    for group, value in (
        (g.batch_labels, "batch"),
        (g.m_labels, "row"),
        (g.n_labels, "column"),
        (g.k_labels, "reduction"),
    ):
        shape = tuple(g.extents[label] for label in group)
        coordinates.update(
            (label, _coordinate(value, shape, axis)) for axis, label in enumerate(group)
        )
    return _flat(
        [coordinates[label] for label in labels],
        tuple(g.extents[label] for label in labels),
    )


def _packing_kernels(
    plan: typing.Any, i: typing.Any, prefix: typing.Any = ""
) -> typing.Any:
    step = plan.steps[i]
    g = gemm_contract(step.node)
    scalar = scalar_type(step.node.spec.dtype)
    ty, mul = scalar.ctype, scalar.intrinsic("mul")
    width = plan.schedule.staging_width
    if width == 1:
        return f"""
__global__ void {_name(prefix, f"pack_{i}")}(const unsigned char* p, {ty}* a, {ty}* b, int* error,
                        I batch, I m0, I n0, I k0, I tm, I tn, I tk) {{
    for (I z = I(blockIdx.x) * blockDim.x + threadIdx.x; z < tm*tk + tk*tn;
         z += I(blockDim.x) * gridDim.x) {{
        if (z < tm*tk) {{
            I row = m0 + z/tk, reduction = k0 + z%tk;
            a[z] = {_read(step.inputs[0], _group_map(g, g.a_labels), prefix)};
        }} else {{
            I q = z - tm*tk;
            I column = n0 + q%tn, reduction = k0 + q/tn;
            b[q] = {_read(step.inputs[1], _group_map(g, g.b_labels), prefix)};
        }}
    }}
}}
__global__ void {_name(prefix, f"scatter_{i}")}(unsigned char* p, const {ty}* c, int* error,
                           I batch, I m0, I n0, I tm, I tn) {{
    for (I z = I(blockIdx.x) * blockDim.x + threadIdx.x; z < tm*tn;
         z += I(blockDim.x) * gridDim.x) {{
        I row = m0 + z/tn, column = n0 + z%tn;
        reinterpret_cast<{ty}*>(p + {step.offset})[{_physical_index(step.layout, _group_map(g, g.output_labels))}] =
            finite({mul}(finite(c[z], error, {i}), {scalar.literal(step.node.attrs["coefficient"])}), error, {i});
    }}
}}
"""
    return f"""
__global__ void {_name(prefix, f"pack_{i}")}(const unsigned char* p, {ty}* a, {ty}* b, int* error,
                        I batch, I m0, I n0, I k0, I tm, I tn, I tk) {{
    const I total = tm*tk + tk*tn;
    for (I base = (I(blockIdx.x) * blockDim.x + threadIdx.x) * {width}LL;
         base < total; base += I(blockDim.x) * gridDim.x * {width}LL) {{
#pragma unroll {width}
        for (int lane = 0; lane < {width}; ++lane) {{
            I z = base + lane;
            if (z >= total) break;
            if (z < tm*tk) {{
                I row = m0 + z/tk, reduction = k0 + z%tk;
                a[z] = {_read(step.inputs[0], _group_map(g, g.a_labels), prefix)};
            }} else {{
                I q = z - tm*tk;
                I column = n0 + q%tn, reduction = k0 + q/tn;
                b[q] = {_read(step.inputs[1], _group_map(g, g.b_labels), prefix)};
            }}
        }}
    }}
}}
__global__ void {_name(prefix, f"scatter_{i}")}(unsigned char* p, const {ty}* c, int* error,
                           I batch, I m0, I n0, I tm, I tn) {{
    const I total = tm*tn;
    for (I base = (I(blockIdx.x) * blockDim.x + threadIdx.x) * {width}LL;
         base < total; base += I(blockDim.x) * gridDim.x * {width}LL) {{
#pragma unroll {width}
        for (int lane = 0; lane < {width}; ++lane) {{
            I z = base + lane;
            if (z >= total) break;
            I row = m0 + z/tn, column = n0 + z%tn;
            reinterpret_cast<{ty}*>(p + {step.offset})[{_physical_index(step.layout, _group_map(g, g.output_labels))}] =
                finite({mul}(finite(c[z], error, {i}), {scalar.literal(step.node.attrs["coefficient"])}), error, {i});
        }}
    }}
}}
"""


def _cooperative_reduce(plan: typing.Any, i: int) -> bool:
    """Whether this step uses either cooperative reduction provider."""

    return cooperative_reduction_provider(plan, i) is not None


def _generated_cooperative_reduce_kernel(
    plan: typing.Any, i: int, prefix: typing.Any = ""
) -> str:
    step = plan.steps[i]
    node = step.node
    scalar = scalar_type(node.spec.dtype)
    accumulator = scalar_type(_value_precision(plan, i).accumulation_dtype)
    acc_add = accumulator.intrinsic("add")
    source_index, reduction_size = _reduce_source_index(node)
    contribution = _convert(
        _read(step.inputs[0], source_index, prefix), scalar, accumulator
    )
    result = _convert("value", accumulator, scalar)
    threads = plan.schedule.threads
    warps = (threads + 31) // 32
    reduction_pragma = (
        ""
        if plan.schedule.reduction_unroll == 1
        else f"#pragma unroll {plan.schedule.reduction_unroll}\n"
    )
    target = _physical_index(step.layout, "z")
    return f"""__global__ void {prefix}kernel_{i}(unsigned char* p, int* error) {{
    __shared__ {accumulator.ctype} partial[{warps}];
    for (I z = I(blockIdx.x); z < {node.spec.size}LL; z += I(gridDim.x)) {{
        {accumulator.ctype} value = {accumulator.zero};
{reduction_pragma}        for (I r = threadIdx.x; r < {_integer(reduction_size)}; r += blockDim.x)
            value = {acc_add}(value, {contribution});
        for (int offset = 16; offset > 0; offset >>= 1)
            value = {acc_add}(value, __shfl_down_sync(0xffffffffu, value, offset));
        const int lane = int(threadIdx.x) & 31;
        const int warp = int(threadIdx.x) >> 5;
        if (lane == 0) partial[warp] = value;
        __syncthreads();
        if (warp == 0) {{
            value = lane < {warps} ? partial[lane] : {accumulator.zero};
            for (int offset = 16; offset > 0; offset >>= 1)
                value = {acc_add}(value, __shfl_down_sync(0xffffffffu, value, offset));
            if (lane == 0)
                reinterpret_cast<{scalar.ctype}*>(p + {step.offset})[{target}] = finite({result}, error, {i});
        }}
        __syncthreads();
    }}
}}"""


def _cub_cooperative_reduce_kernel(
    plan: typing.Any, i: int, prefix: typing.Any = ""
) -> str:
    step = plan.steps[i]
    node = step.node
    scalar = scalar_type(node.spec.dtype)
    accumulator = scalar_type(_value_precision(plan, i).accumulation_dtype)
    acc_add = accumulator.intrinsic("add")
    source_index, reduction_size = _reduce_source_index(node)
    contribution = _convert(
        _read(step.inputs[0], source_index, prefix), scalar, accumulator
    )
    result = _convert("value", accumulator, scalar)
    threads = plan.schedule.threads
    reduction_pragma = (
        ""
        if plan.schedule.reduction_unroll == 1
        else f"#pragma unroll {plan.schedule.reduction_unroll}\n"
    )
    target = _physical_index(step.layout, "z")
    add_name = _name(prefix, f"cub_add_{i}")
    return f"""struct {add_name} {{
    __device__ __forceinline__ {accumulator.ctype} operator()(
        {accumulator.ctype} a, {accumulator.ctype} b) const {{
        return {acc_add}(a, b);
    }}
}};
__global__ void {prefix}kernel_{i}(unsigned char* p, int* error) {{
    using BlockReduce = cub::BlockReduce<
        {accumulator.ctype}, {threads}, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
    using TempStorage = typename BlockReduce::TempStorage;
    __shared__ TempStorage temp_storage;
    for (I z = I(blockIdx.x); z < {node.spec.size}LL; z += I(gridDim.x)) {{
        {accumulator.ctype} value = {accumulator.zero};
{reduction_pragma}        for (I r = threadIdx.x; r < {_integer(reduction_size)}; r += blockDim.x)
            value = {acc_add}(value, {contribution});
        value = BlockReduce(temp_storage).Reduce(value, {add_name}{{}});
        if (threadIdx.x == 0)
            reinterpret_cast<{scalar.ctype}*>(p + {step.offset})[{target}] =
                finite({result}, error, {i});
        __syncthreads();
    }}
}}"""


def _cooperative_reduce_kernel(
    plan: typing.Any, i: int, prefix: typing.Any = ""
) -> str:
    provider = cooperative_reduction_provider(plan, i)
    if provider == "generated":
        return _generated_cooperative_reduce_kernel(plan, i, prefix)
    if provider == "cub":
        return _cub_cooperative_reduce_kernel(plan, i, prefix)
    raise ValueError("cooperative reduction kernel requested for an ordinary step")


def _launch(plan: typing.Any, i: typing.Any, prefix: typing.Any = "") -> typing.Any:
    step, threads = plan.steps[i], plan.schedule.threads
    node = step.node
    if step.virtual or node.op in ("input", "constant") or not node.spec.size:
        return ""
    scalar = scalar_type(node.spec.dtype)
    ty = scalar.ctype
    pointer = f"reinterpret_cast<{ty}*>(p + {step.offset})"
    if _cooperative_reduce(plan, i):
        return f"ctx.section(profile, metrics.kernel_ms, [&] {{ {prefix}kernel_{i}<<<blocks({node.spec.size}LL, 1), {threads}, 0, ctx.stream>>>(p, ctx.error); cuda_check(cudaGetLastError()); }});"
    if step.gemm == "none":
        width = plan.schedule.elements_per_thread
        work_items = (node.spec.size + width - 1) // width
        return f"ctx.section(profile, metrics.kernel_ms, [&] {{ {prefix}kernel_{i}<<<blocks({work_items}LL, {threads}), {threads}, 0, ctx.stream>>>(p, ctx.error); cuda_check(cudaGetLastError()); }});"
    g = gemm_contract(node)
    if not g.k:
        return f"ctx.section(profile, metrics.kernel_ms, [&] {{ cuda_check(cudaMemsetAsync({pointer}, 0, {node.spec.size * node.spec.itemsize}ULL, ctx.stream)); }});"
    if step.gemm.startswith("direct-"):
        a, b = [
            f"reinterpret_cast<const {ty}*>(p + {plan.steps[c].offset})"
            for c in step.inputs
        ]
        ta, tb = step.gemm[-2:]
        return f"""
ctx.section(profile, metrics.library_ms, [&] {{
    gemm(ctx, '{ta}', '{tb}', {g.m}, {g.n}, {g.k}, {a}, {b}, {pointer},
         {g.m * g.k}LL, {g.k * g.n}LL, {g.m * g.n}LL, {g.batch}, {scalar.zero});
}});
ctx.section(profile, metrics.kernel_ms, [&] {{
    check_scale<<<blocks({node.spec.size}LL, {threads}), {threads}, 0, ctx.stream>>>({pointer}, {node.spec.size}LL, {scalar.literal(node.attrs["coefficient"])}, ctx.error, {i});
    cuda_check(cudaGetLastError());
}});"""
    mt, nt, kt = [
        min(tile, size)
        for tile, size in zip(
            (plan.schedule.tile_m, plan.schedule.tile_n, plan.schedule.tile_k),
            (g.m, g.n, g.k),
            strict=True,
        )
    ]
    staging_width = plan.schedule.staging_width
    pack_work = (
        "tm*tk+tk*tn"
        if staging_width == 1
        else f"(tm*tk+tk*tn+{staging_width - 1}LL)/{staging_width}LL"
    )
    scatter_work = (
        "tm*tn"
        if staging_width == 1
        else f"(tm*tn+{staging_width - 1}LL)/{staging_width}LL"
    )
    return f"""{{
{ty}* a = reinterpret_cast<{ty}*>(p + {plan.arena_bytes});
{ty}* b = a + {mt * kt}LL;
{ty}* c = b + {kt * nt}LL;
for (I batch = 0; batch < {g.batch}LL; ++batch)
for (I m0 = 0; m0 < {g.m}LL; m0 += {mt}LL)
for (I n0 = 0; n0 < {g.n}LL; n0 += {nt}LL) {{
    I tm = std::min<I>({mt}, {g.m}LL-m0), tn = std::min<I>({nt}, {g.n}LL-n0);
    for (I k0 = 0; k0 < {g.k}LL; k0 += {kt}LL) {{
        I tk = std::min<I>({kt}, {g.k}LL-k0);
        ctx.section(profile, metrics.packing_ms, [&] {{
            {_name(prefix, f"pack_{i}")}<<<blocks({pack_work}, {threads}), {threads}, 0, ctx.stream>>>(p, a, b, ctx.error, batch, m0, n0, k0, tm, tn, tk);
            cuda_check(cudaGetLastError());
        }});
        ctx.section(profile, metrics.library_ms, [&] {{ gemm(ctx, 'N', 'N', int(tm), int(tn), int(tk), a, b, c, 0, 0, 0, 1, k0 == 0 ? {scalar.zero} : {scalar.one}); }});
    }}
    ctx.section(profile, metrics.packing_ms, [&] {{
        {_name(prefix, f"scatter_{i}")}<<<blocks({scatter_work}, {threads}), {threads}, 0, ctx.stream>>>(p, c, ctx.error, batch, m0, n0, tm, tn);
        cuda_check(cudaGetLastError());
    }});
}}
}}"""


def emit_cuda(
    plan: TensorPlan, symbol_prefix: str = "", *, embed_static_data: bool = True
) -> str:
    """Return standalone C++17 CUDA source with an optional symbol prefix.

    A prefix places the generated ABI in a unique namespace and prefixes all
    helper and entry-point names. The default keeps the original standalone
    ``tensor_*`` interface unchanged.
    """
    import re

    for step in plan.steps:
        if not step.virtual and (
            step.layout is None or step.layout.shape != step.node.spec.shape
        ):
            raise ValueError("materialized tensor layout must match its logical shape")
    for i in (*plan.inputs, *(index for _, index in plan.outputs)):
        if plan.steps[i].virtual or not plan.steps[i].layout.is_c_contiguous:
            raise ValueError("tensor ABI inputs and outputs must use logical C-order")

    if not isinstance(symbol_prefix, str) or (
        symbol_prefix
        and not re.fullmatch(r"[A-Za-z_]\w*", symbol_prefix, flags=re.ASCII)
    ):
        raise ValueError("symbol_prefix must be a valid C++ identifier")
    prefix = symbol_prefix
    namespace = f"namespace {_name(prefix, 'generated')} {{" if prefix else ""
    parts = ['#include "cuda_graph_context.cuh"']
    if any(
        cooperative_reduction_provider(plan, i) == "cub" for i in range(len(plan.steps))
    ):
        parts.append("#include <cub/block/block_reduce.cuh>")
    parts.append("using namespace vibeqc_tensor;")
    if namespace:
        parts.append(namespace)
    dtypes = sorted({step.node.spec.dtype for step in plan.steps})
    if "float32" in dtypes:
        parts.append(
            "#if defined(__CUDA_FTZ) && __CUDA_FTZ\n#error FP32 TensorIR requires --ftz=false\n#endif"
        )
    for dtype in dtypes:
        if any(
            s.node.op == "scaled_bilinear" and s.node.spec.dtype == dtype
            for s in plan.steps
        ):
            parts.append(emit_scaled_bilinear(prefix, dtype=dtype))
    initialize = []
    tables = dict(plan.index_tables)
    for i, step in enumerate(plan.steps):
        node = step.node
        scalar = None if node.spec.dtype == "int64" else scalar_type(node.spec.dtype)
        ty = "I" if scalar is None else scalar.ctype
        if embed_static_data and node.op == "constant" and node.spec.size:
            values = ", ".join(scalar.literal(pair) for pair in node.attrs["values"])
            parts.append(f"static const {ty} {prefix}constant_{i}[] = {{{values}}};")
            initialize.append(
                f"cuda_check(cudaMemcpyAsync(ctx->arena + {step.offset}, {prefix}constant_{i}, {node.spec.size * node.spec.itemsize}ULL, cudaMemcpyHostToDevice, ctx->stream));"
            )
        table_values = _index_table_values(node)
        if embed_static_data and table_values:
            values = ", ".join(_integer(v) for v in table_values)
            parts.append(f"static const I {prefix}index_data_{i}[] = {{{values}}};")
            initialize.append(
                f"cuda_check(cudaMemcpyAsync(ctx->arena + {tables[i]}, {prefix}index_data_{i}, {len(table_values) * 8}ULL, cudaMemcpyHostToDevice, ctx->stream));"
            )
        body = (
            _value(plan, i, prefix)
            if step.virtual
            else f"return reinterpret_cast<const {ty}*>(p + {step.offset})[{_physical_index(step.layout)}];"
        )
        parts.append(
            f"__device__ inline {ty} {prefix}read_{i}(const unsigned char* p, I z, int* error) {{ {body} }}"
        )
        if not step.virtual and node.op not in ("input", "constant"):
            if step.gemm == "none":
                if _cooperative_reduce(plan, i):
                    parts.append(_cooperative_reduce_kernel(plan, i, prefix))
                    continue
                width = plan.schedule.elements_per_thread
                if width == 1:
                    parts.append(f"""__device__ inline {ty} {prefix}evaluate_{i}(const unsigned char* p, I z, int* error) {{ {_value(plan, i, prefix)} }}
__global__ void {prefix}kernel_{i}(unsigned char* p, int* error) {{
    for (I z = I(blockIdx.x) * blockDim.x + threadIdx.x; z < {node.spec.size}LL;
         z += I(blockDim.x) * gridDim.x)
        reinterpret_cast<{ty}*>(p + {step.offset})[z] = {prefix}evaluate_{i}(p, {_logical_index(step.layout)}, error);
}}""")
                else:
                    parts.append(f"""__device__ inline {ty} {prefix}evaluate_{i}(const unsigned char* p, I z, int* error) {{ {_value(plan, i, prefix)} }}
__global__ void {prefix}kernel_{i}(unsigned char* p, int* error) {{
    for (I base = (I(blockIdx.x) * blockDim.x + threadIdx.x) * {width}LL;
         base < {node.spec.size}LL; base += I(blockDim.x) * gridDim.x * {width}LL) {{
#pragma unroll {width}
        for (int lane = 0; lane < {width}; ++lane) {{
            I z = base + lane;
            if (z >= {node.spec.size}LL) break;
            reinterpret_cast<{ty}*>(p + {step.offset})[z] = {prefix}evaluate_{i}(p, {_logical_index(step.layout)}, error);
        }}
    }}
}}""")
            elif step.gemm == "packed":
                parts.append(_packing_kernels(plan, i, prefix))
    copies_in = []
    for slot, i in enumerate(plan.inputs):
        step = plan.steps[i]
        if step.node.spec.size:
            copies_in.append(
                f'if (!inputs[{slot}]) throw std::runtime_error("null tensor input");\ncuda_check(cudaMemcpyAsync(p + {step.offset}, inputs[{slot}], {step.node.spec.size * step.node.spec.itemsize}ULL, cudaMemcpyHostToDevice, ctx.stream));'
            )
    copies_out = []
    for slot, (_, i) in enumerate(plan.outputs):
        step = plan.steps[i]
        if step.node.spec.size:
            copies_out.append(
                f'if (!outputs[{slot}]) throw std::runtime_error("null tensor output");\ncuda_check(cudaMemcpyAsync(outputs[{slot}], p + {step.offset}, {step.node.spec.size * step.node.spec.itemsize}ULL, cudaMemcpyDeviceToHost, ctx.stream));'
            )
    needs_blas = any(
        s.gemm != "none" and gemm_contract(s.node).k and s.node.spec.size
        for s in plan.steps
    )
    fp32_blas = any(
        s.gemm != "none" and s.node.spec.dtype == "float32" for s in plan.steps
    )
    math_mode = (
        "if (ctx->handle) blas_check(cublasSetMathMode(ctx->handle, CUBLAS_PEDANTIC_MATH));"
        if fp32_blas
        else ""
    )
    library_offset = plan.arena_bytes + plan.panel_bytes
    error_offset = (
        library_offset + plan.library_bytes + aligned(plan.reservations.total)
    )
    assert error_offset + ALIGNMENT == plan.allocation_bytes
    error_expression = _arithmetic_error_expression(
        plan,
        'std::string(arithmetic_error < 0 ? "tensor division by zero at step " : "non-finite tensor at step ") + std::to_string(std::abs(arithmetic_error)-1)',
    )
    external_slices = () if embed_static_data else static_data_slices(plan)
    external_static_bytes = sum(item[4] for item in external_slices)
    external_copies = " ".join(
        f"cuda_check(cudaMemcpyAsync(ctx.arena + {arena_offset}, bytes + {payload_offset}, {size_bytes}ULL, cudaMemcpyHostToDevice, ctx.stream));"
        for _, _, arena_offset, payload_offset, size_bytes in external_slices
    )
    static_abi = ""
    if not embed_static_data:
        static_abi = f"""
extern "C" size_t {_name(prefix, "tensor_static_bytes")}() {{ return {external_static_bytes}ULL; }}
extern "C" int {_name(prefix, "tensor_static_initialize")}(void* pointer, const void* data, size_t bytes_count,
                          char* error, size_t size) {{
    if (!pointer) {{ error_text(error, size, "null tensor plan"); return 1; }}
    auto& ctx = *static_cast<GraphContext*>(static_cast<Context*>(pointer));
    std::unique_lock<std::mutex> lock(ctx.mutex, std::try_to_lock);
    if (!lock.owns_lock()) {{ error_text(error, size, "tensor plan is already executing"); return 1; }}
    bool uploading = false;
    try {{
        ctx.check_device();
        if (ctx.static_ready)
            throw std::runtime_error("tensor static data is already initialized");
        if (bytes_count != {external_static_bytes}ULL)
            throw std::runtime_error("tensor static-data size mismatch");
        if (bytes_count && !data)
            throw std::runtime_error("null tensor static-data payload");
        const auto* bytes = static_cast<const unsigned char*>(data);
        uploading = true;
        {external_copies}
        cuda_check(cudaStreamSynchronize(ctx.stream));
        ctx.static_ready = true;
        return 0;
    }} catch (const std::exception& e) {{
        // A queued copy still borrows data even if a later submission failed.
        // Drain before the caller can release the bounded host payload.
        if (uploading) cudaStreamSynchronize(ctx.stream);
        error_text(error, size, e.what()); return 1;
    }}
}}
"""
    parts.append(f"""
extern "C" const char* {_name(prefix, "tensor_plan_identity")}() {{ return "{plan.identity}"; }}
extern "C" int {_name(prefix, "tensor_create")}(int device, void** result, char* error, size_t size) {{
    try {{
        if (!result) throw std::runtime_error("null plan output");
        *result = nullptr;
        auto ctx = std::make_unique<GraphContext>();
        ctx->prepare(device, {plan.target.compute_capability_major}, {plan.target.compute_capability_minor},
                     {plan.allocation_bytes}ULL, {error_offset}ULL, {library_offset}ULL,
                     {plan.library_bytes}ULL, {plan.provider_bytes}ULL, {"true" if needs_blas else "false"});
        vibeqc::runtime::CudaDeviceScope guard(device, cuda_check);
        {math_mode}
        {"ctx->static_ready = false;" if not embed_static_data else ""}
        {" ".join(initialize)}
        cuda_check(cudaStreamSynchronize(ctx->stream));
        *result = static_cast<Context*>(ctx.release());
        return 0;
    }} catch (const DeviceAllocationError& e) {{
        error_text(error, size, e.what()); return 2;
    }} catch (const std::bad_alloc& e) {{
        error_text(error, size, e.what()); return 3;
    }} catch (const std::exception& e) {{ error_text(error, size, e.what()); return 1; }}
}}
extern "C" void {_name(prefix, "tensor_destroy")}(void* pointer) {{ delete static_cast<GraphContext*>(static_cast<Context*>(pointer)); }}
{static_abi}
static int {_name(prefix, "tensor_run_impl")}(void* pointer, const void* const* inputs, void* const* outputs,
                          int profile, Metrics* result, vibeqc::runtime::GraphMetrics* graph_result,
                          char* graph_reason, size_t graph_reason_size, char* error, size_t size) {{
    if (!pointer) {{ error_text(error, size, "null tensor plan"); return 1; }}
    auto& ctx = *static_cast<GraphContext*>(static_cast<Context*>(pointer));
    std::unique_lock<std::mutex> lock(ctx.mutex, std::try_to_lock);
    if (!lock.owns_lock()) {{ error_text(error, size, "tensor plan is already executing"); return 1; }}
    try {{
        ctx.check_device();
        if (!ctx.static_ready) throw std::runtime_error("tensor static data is not initialized");
        if (!result || !inputs || !outputs) throw std::runtime_error("null tensor execution arguments");
        Metrics metrics;
        metrics.owned_device_bytes = ctx.metrics.owned_device_bytes;
        metrics.provider_retained_bytes = ctx.metrics.provider_retained_bytes;
        metrics.prepare_device_delta = ctx.metrics.prepare_device_delta;
        auto* p = ctx.arena;
        cuda_check(cudaEventRecord(ctx.begin, ctx.stream));
        ctx.section(profile, metrics.input_ms, [&] {{ {" ".join(copies_in)} }});
        ctx.submit_region(profile, [&] {{
            cuda_check(cudaMemsetAsync(ctx.error, 0, sizeof(int), ctx.stream));
            {" ".join(_launch(plan, i, prefix) for i in range(len(plan.steps)))}
        }});
        int arithmetic_error = 0;
        ctx.section(profile, metrics.output_ms, [&] {{
            {" ".join(copies_out)}
            cuda_check(cudaMemcpyAsync(&arithmetic_error, ctx.error, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream));
        }});
        cuda_check(cudaEventRecord(ctx.end, ctx.stream));
        cuda_check(cudaEventSynchronize(ctx.end));
        float elapsed = 0;
        cuda_check(cudaEventElapsedTime(&elapsed, ctx.begin, ctx.end));
        metrics.device_ms = elapsed;
        metrics.observed_device_delta = std::max(ctx.metrics.prepare_device_delta, ctx.device_delta());
        *result = metrics;
        if (graph_result) *graph_result = ctx.graph.metrics;
        error_text(graph_reason, graph_reason_size, ctx.graph.reason.c_str());
        if (arithmetic_error)
            throw std::runtime_error({error_expression});
        return 0;
    }} catch (const std::exception& e) {{
        // Drain queued host transfers before Python may release their arrays.
        cudaStreamSynchronize(ctx.stream);
        error_text(error, size, e.what()); return 1;
    }}
}}
extern "C" int {_name(prefix, "tensor_run")}(void* pointer, const void* const* inputs, void* const* outputs,
                          int profile, Metrics* result, char* error, size_t size) {{
    return {_name(prefix, "tensor_run_impl")}(pointer, inputs, outputs, profile, result, nullptr, nullptr, 0, error, size);
}}
extern "C" int {_name(prefix, "tensor_run_graph")}(void* pointer, const void* const* inputs, void* const* outputs,
                          int profile, Metrics* result, vibeqc::runtime::GraphMetrics* graph_result,
                          char* reason, size_t reason_size, char* error, size_t size) {{
    return {_name(prefix, "tensor_run_impl")}(pointer, inputs, outputs, profile, result, graph_result, reason, reason_size, error, size);
}}
extern "C" int {_name(prefix, "tensor_graph_configure")}(void* pointer, int enabled, const char* identity,
                          char* error, size_t size) {{
    if (!pointer) {{ error_text(error, size, "null tensor plan"); return 1; }}
    auto& ctx = *static_cast<GraphContext*>(static_cast<Context*>(pointer));
    std::unique_lock<std::mutex> lock(ctx.mutex, std::try_to_lock);
    if (!lock.owns_lock()) {{ error_text(error, size, "tensor plan is already executing"); return 1; }}
    try {{ ctx.configure_graph(enabled != 0, identity); return 0; }}
    catch (const std::exception& e) {{ error_text(error, size, e.what()); return 1; }}
}}
extern "C" int {_name(prefix, "tensor_probe")}(int device, char* result, size_t size) {{
    try {{
        vibeqc::runtime::CudaDeviceScope guard(device, cuda_check);
        cudaDeviceProp p{{}};
        cuda_check(cudaGetDeviceProperties(&p, device));
        int driver = 0, runtime = 0, major = 0, minor = 0, patch = 0;
        cuda_check(cudaDriverGetVersion(&driver));
        cuda_check(cudaRuntimeGetVersion(&runtime));
        blas_check(cublasGetProperty(MAJOR_VERSION, &major));
        blas_check(cublasGetProperty(MINOR_VERSION, &minor));
        blas_check(cublasGetProperty(PATCH_LEVEL, &patch));
        char uuid[33];
        for (int i = 0; i < 16; ++i) std::snprintf(uuid+2*i, 3, "%02x", static_cast<unsigned char>(p.uuid.bytes[i]));
        std::snprintf(result, size,
            "{{\\"uuid\\":\\"%s\\",\\"architecture\\":\\"sm_%d%d\\",\\"sm_count\\":%d,\\"driver\\":%d,\\"runtime\\":%d,\\"cublas\\":\\"%d.%d.%d\\"}}",
            uuid, p.major, p.minor, p.multiProcessorCount, driver, runtime, major, minor, patch);
        return 0;
    }} catch (const std::exception& e) {{ error_text(result, size, e.what()); return 1; }}
}}
""")
    if namespace:
        parts.append("}")
    return "\n\n".join(parts) + "\n"
