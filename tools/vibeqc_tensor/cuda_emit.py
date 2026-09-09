"""Emit one native CUDA executor per immutable TensorIR execution plan.

Python generates the whole launch/contraction sequence ahead of time. Runtime
host code only moves data and schedules device work; all tensor arithmetic,
including general einsums and denominators, executes on the allocated GPU.
"""

from __future__ import annotations

from math import prod

from .cuda_gemm import fp64_coefficient, gemm_contract
from .cuda_plan import ALIGNMENT, TensorPlan, aligned, strides


def _integer(value):
    return f"{value}LL"


def _coordinate(linear, shape, axis):
    # Empty kernels/accessors are never executed, but CUDA still compiles
    # their bodies. Avoid constant division by zero even in unreachable code.
    return f"(({linear}) / {_integer(max(1, prod(shape[axis + 1 :])))} % {_integer(max(1, shape[axis]))})"


def _flat(coordinates, shape):
    return (
        " + ".join(
            f"({coord}) * {_integer(stride)}"
            for coord, stride in zip(coordinates, strides(shape), strict=True)
        )
        or "0LL"
    )


def _read(operand, index):
    return f"read_{operand}(p, {index}, error)"


def _value(plan, i):
    """Emit scalar evaluation with each original arithmetic error boundary."""
    step = plan.steps[i]
    node, a, args = step.node, step.node.attrs, step.inputs
    shape = node.spec.shape
    c = [_coordinate("z", shape, axis) for axis in range(len(shape))]
    if node.op == "add":
        lines = ["double value = 0.0;"]
        for child, factor in zip(args, a["coefficients"], strict=True):
            lines.append(
                f"value = __dadd_rn(value, __dmul_rn({fp64_coefficient(factor).hex()}, {_read(child, 'z')}));"
            )
        return "\n".join(lines + [f"return finite(value, error, {i});"])
    if node.op == "multiply":
        return f"return finite(__dmul_rn({_read(args[0], 'z')}, {_read(args[1], 'z')}), error, {i});"
    if node.op == "divide":
        return f"return quotient({_read(args[0], 'z')}, {_read(args[1], 'z')}, error, {i});"
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
            )
            for child, labels in zip(args, a["labels"], strict=True)
        ]
        term = values[0]
        for value in values[1:]:
            term = f"__dmul_rn({term}, {value})"
        return f"""double value = 0.0;
for (I r = 0; r < {_integer(prod(reduction_shape))}; ++r)
    value = __dadd_rn(value, {term});
return finite(__dmul_rn(finite(value, error, {i}), {fp64_coefficient(a["coefficient"]).hex()}), error, {i});"""
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
    elif node.op == "gather":
        table = dict(plan.index_tables)[i]
        c[a["axis"]] = f"reinterpret_cast<const I*>(p + {table})[{c[a['axis']]}]"
        index = _flat(c, source_shape)
    elif node.op == "reduce":
        reduction_shape = tuple(source_shape[axis] for axis in a["axes"])
        source, cursor = [], 0
        for axis in range(len(source_shape)):
            if axis in a["axes"]:
                source.append(_coordinate("r", reduction_shape, a["axes"].index(axis)))
            else:
                source.append(c[cursor])
                cursor += 1
        return f"""double value = 0.0;
for (I r = 0; r < {_integer(prod(reduction_shape))}; ++r)
    value = __dadd_rn(value, {_read(child, _flat(source, source_shape))});
return finite(value, error, {i});"""
    else:
        raise ValueError(f"unsupported CUDA primitive: {node.op}")
    return f"return {_read(child, index)};"


def _group_map(g, labels):
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


def _packing_kernels(plan, i):
    step = plan.steps[i]
    g = gemm_contract(step.node)
    return f"""
__global__ void pack_{i}(const unsigned char* p, double* a, double* b, int* error,
                        I batch, I m0, I n0, I k0, I tm, I tn, I tk) {{
    for (I z = I(blockIdx.x) * blockDim.x + threadIdx.x; z < tm*tk + tk*tn;
         z += I(blockDim.x) * gridDim.x) {{
        if (z < tm*tk) {{
            I row = m0 + z/tk, reduction = k0 + z%tk;
            a[z] = {_read(step.inputs[0], _group_map(g, g.a_labels))};
        }} else {{
            I q = z - tm*tk;
            I column = n0 + q%tn, reduction = k0 + q/tn;
            b[q] = {_read(step.inputs[1], _group_map(g, g.b_labels))};
        }}
    }}
}}
__global__ void scatter_{i}(unsigned char* p, const double* c, int* error,
                           I batch, I m0, I n0, I tm, I tn) {{
    for (I z = I(blockIdx.x) * blockDim.x + threadIdx.x; z < tm*tn;
         z += I(blockDim.x) * gridDim.x) {{
        I row = m0 + z/tn, column = n0 + z%tn;
        reinterpret_cast<double*>(p + {step.offset})[{_group_map(g, g.output_labels)}] =
            finite(__dmul_rn(finite(c[z], error, {i}), {g.coefficient.hex()}), error, {i});
    }}
}}
"""


def _launch(plan, i):
    step, threads = plan.steps[i], plan.schedule.threads
    node = step.node
    if step.virtual or node.op in ("input", "constant") or not node.spec.size:
        return ""
    pointer = f"reinterpret_cast<double*>(p + {step.offset})"
    if step.gemm == "none":
        return f"ctx.section(profile, metrics.kernel_ms, [&] {{ kernel_{i}<<<blocks({node.spec.size}LL, {threads}), {threads}, 0, ctx.stream>>>(p, ctx.error); cuda_check(cudaGetLastError()); }});"
    g = gemm_contract(node)
    if not g.k:
        return f"ctx.section(profile, metrics.kernel_ms, [&] {{ cuda_check(cudaMemsetAsync({pointer}, 0, {node.spec.size * 8}ULL, ctx.stream)); }});"
    if step.gemm.startswith("direct-"):
        a, b = [
            f"reinterpret_cast<const double*>(p + {plan.steps[c].offset})"
            for c in step.inputs
        ]
        ta, tb = step.gemm[-2:]
        return f"""
ctx.section(profile, metrics.library_ms, [&] {{
    gemm(ctx, '{ta}', '{tb}', {g.m}, {g.n}, {g.k}, {a}, {b}, {pointer},
         {g.m * g.k}LL, {g.k * g.n}LL, {g.m * g.n}LL, {g.batch}, 0.0);
}});
ctx.section(profile, metrics.kernel_ms, [&] {{
    check_scale<<<blocks({node.spec.size}LL, {threads}), {threads}, 0, ctx.stream>>>({pointer}, {node.spec.size}LL, {g.coefficient.hex()}, ctx.error, {i});
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
    return f"""{{
double* a = reinterpret_cast<double*>(p + {plan.arena_bytes});
double* b = a + {mt * kt}LL;
double* c = b + {kt * nt}LL;
for (I batch = 0; batch < {g.batch}LL; ++batch)
for (I m0 = 0; m0 < {g.m}LL; m0 += {mt}LL)
for (I n0 = 0; n0 < {g.n}LL; n0 += {nt}LL) {{
    I tm = std::min<I>({mt}, {g.m}LL-m0), tn = std::min<I>({nt}, {g.n}LL-n0);
    for (I k0 = 0; k0 < {g.k}LL; k0 += {kt}LL) {{
        I tk = std::min<I>({kt}, {g.k}LL-k0);
        ctx.section(profile, metrics.packing_ms, [&] {{
            pack_{i}<<<blocks(tm*tk+tk*tn, {threads}), {threads}, 0, ctx.stream>>>(p, a, b, ctx.error, batch, m0, n0, k0, tm, tn, tk);
            cuda_check(cudaGetLastError());
        }});
        ctx.section(profile, metrics.library_ms, [&] {{ gemm(ctx, 'N', 'N', int(tm), int(tn), int(tk), a, b, c, 0, 0, 0, 1, k0 == 0 ? 0.0 : 1.0); }});
    }}
    ctx.section(profile, metrics.packing_ms, [&] {{
        scatter_{i}<<<blocks(tm*tn, {threads}), {threads}, 0, ctx.stream>>>(p, c, ctx.error, batch, m0, n0, tm, tn);
        cuda_check(cudaGetLastError());
    }});
}}
}}"""


def emit_cuda(plan: TensorPlan) -> str:
    """Return standalone C++17 CUDA source with a versioned, exception-safe ABI."""
    parts = ['#include "cuda_runtime.cuh"', "using namespace vibeqc_tensor;"]
    initialize = []
    tables = dict(plan.index_tables)
    for i, step in enumerate(plan.steps):
        node = step.node
        if node.op == "constant" and node.spec.size:
            values = ", ".join(
                fp64_coefficient(pair).hex() for pair in node.attrs["values"]
            )
            parts.append(f"static const double constant_{i}[] = {{{values}}};")
            initialize.append(
                f"cuda_check(cudaMemcpyAsync(ctx->arena + {step.offset}, constant_{i}, {node.spec.size * 8}ULL, cudaMemcpyHostToDevice, ctx->stream));"
            )
        if node.op == "gather" and node.attrs["positions"]:
            values = ", ".join(_integer(v) for v in node.attrs["positions"])
            parts.append(f"static const I positions_{i}[] = {{{values}}};")
            initialize.append(
                f"cuda_check(cudaMemcpyAsync(ctx->arena + {tables[i]}, positions_{i}, {len(node.attrs['positions']) * 8}ULL, cudaMemcpyHostToDevice, ctx->stream));"
            )
        body = (
            _value(plan, i)
            if step.virtual
            else f"return reinterpret_cast<const double*>(p + {step.offset})[z];"
        )
        parts.append(
            f"__device__ inline double read_{i}(const unsigned char* p, I z, int* error) {{ {body} }}"
        )
        if not step.virtual and node.op not in ("input", "constant"):
            if step.gemm == "none":
                parts.append(f"""__device__ inline double evaluate_{i}(const unsigned char* p, I z, int* error) {{ {_value(plan, i)} }}
__global__ void kernel_{i}(unsigned char* p, int* error) {{
    for (I z = I(blockIdx.x) * blockDim.x + threadIdx.x; z < {node.spec.size}LL;
         z += I(blockDim.x) * gridDim.x)
        reinterpret_cast<double*>(p + {step.offset})[z] = evaluate_{i}(p, z, error);
}}""")
            elif step.gemm == "packed":
                parts.append(_packing_kernels(plan, i))
    copies_in = []
    for slot, i in enumerate(plan.inputs):
        step = plan.steps[i]
        if step.node.spec.size:
            copies_in.append(
                f'if (!inputs[{slot}]) throw std::runtime_error("null tensor input");\ncuda_check(cudaMemcpyAsync(p + {step.offset}, inputs[{slot}], {step.node.spec.size * 8}ULL, cudaMemcpyHostToDevice, ctx.stream));'
            )
    copies_out = []
    for slot, (_, i) in enumerate(plan.outputs):
        step = plan.steps[i]
        if step.node.spec.size:
            copies_out.append(
                f'if (!outputs[{slot}]) throw std::runtime_error("null tensor output");\ncuda_check(cudaMemcpyAsync(outputs[{slot}], p + {step.offset}, {step.node.spec.size * 8}ULL, cudaMemcpyDeviceToHost, ctx.stream));'
            )
    needs_blas = any(
        s.gemm != "none" and gemm_contract(s.node).k and s.node.spec.size
        for s in plan.steps
    )
    library_offset = plan.arena_bytes + plan.panel_bytes
    error_offset = (
        library_offset + plan.library_bytes + aligned(plan.reservations.total)
    )
    assert error_offset + ALIGNMENT == plan.allocation_bytes
    parts.append(f"""
extern "C" const char* tensor_plan_identity() {{ return "{plan.identity}"; }}
extern "C" int tensor_create(int device, void** result, char* error, size_t size) {{
    try {{
        if (!result) throw std::runtime_error("null plan output");
        *result = nullptr;
        auto ctx = std::make_unique<Context>();
        ctx->prepare(device, {plan.target.compute_capability_major}, {plan.target.compute_capability_minor},
                     {plan.allocation_bytes}ULL, {error_offset}ULL, {library_offset}ULL,
                     {plan.library_bytes}ULL, {plan.provider_bytes}ULL, {"true" if needs_blas else "false"});
        DeviceGuard guard(device);
        {" ".join(initialize)}
        cuda_check(cudaStreamSynchronize(ctx->stream));
        *result = ctx.release();
        return 0;
    }} catch (const DeviceAllocationError& e) {{
        error_text(error, size, e.what()); return 2;
    }} catch (const std::bad_alloc& e) {{
        error_text(error, size, e.what()); return 3;
    }} catch (const std::exception& e) {{ error_text(error, size, e.what()); return 1; }}
}}
extern "C" void tensor_destroy(void* pointer) {{ delete static_cast<Context*>(pointer); }}
extern "C" int tensor_run(void* pointer, const double* const* inputs, double* const* outputs,
                          int profile, Metrics* result, char* error, size_t size) {{
    if (!pointer) {{ error_text(error, size, "null tensor plan"); return 1; }}
    auto& ctx = *static_cast<Context*>(pointer);
    std::unique_lock<std::mutex> lock(ctx.mutex, std::try_to_lock);
    if (!lock.owns_lock()) {{ error_text(error, size, "tensor plan is already executing"); return 1; }}
    try {{
        ctx.check_device();
        if (!result || !inputs || !outputs) throw std::runtime_error("null tensor execution arguments");
        Metrics metrics;
        metrics.owned_device_bytes = ctx.metrics.owned_device_bytes;
        metrics.provider_retained_bytes = ctx.metrics.provider_retained_bytes;
        metrics.prepare_device_delta = ctx.metrics.prepare_device_delta;
        auto* p = ctx.arena;
        cuda_check(cudaEventRecord(ctx.begin, ctx.stream));
        cuda_check(cudaMemsetAsync(ctx.error, 0, sizeof(int), ctx.stream));
        ctx.section(profile, metrics.input_ms, [&] {{ {" ".join(copies_in)} }});
        {" ".join(_launch(plan, i) for i in range(len(plan.steps)))}
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
        if (arithmetic_error)
            throw std::runtime_error(std::string(arithmetic_error < 0 ? "tensor division by zero at step " : "non-finite tensor at step ") + std::to_string(std::abs(arithmetic_error)-1));
        return 0;
    }} catch (const std::exception& e) {{
        // Drain queued host transfers before Python may release their arrays.
        cudaStreamSynchronize(ctx.stream);
        error_text(error, size, e.what()); return 1;
    }}
}}
extern "C" int tensor_probe(int device, char* result, size_t size) {{
    try {{
        DeviceGuard guard(device);
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
    return "\n\n".join(parts) + "\n"
