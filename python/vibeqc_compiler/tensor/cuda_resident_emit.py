"""Resident ABI appended to the verified #146 generated translation unit.

No generated-source replacement or second mathematical lowering: this module
wraps the complete TU produced by
:func:`vibeqc_compiler.tensor.cuda_emit.emit_cuda` and appends span-based
upload/run/download entry points around the same ``Context`` owner.

The ordinary ``tensor_run`` performs explicit H2D/D2H staging copies at every
call — device-resident iteration cannot use it.  *resident_run* therefore
inlines the generated kernel launch sequence directly: the same mutex guard,
begin/end events, per-section timers, gemm contractions, error-memset and
arithmetic-error readback, but *without* the input/output staging copies.
Inputs are uploaded once through ``resident_upload``; outputs are downloaded
on demand through ``resident_download``.  Every kernel, coefficient, tensor
offset, alignment and error boundary is identical to the ordinary path — only
the per-call transfers are removed.

Physical spans are the plan's pinned materialized step offsets, emitted to
match exactly one named slot.  ``extension`` may add plan-specific tables and
kernels; when it defines ``vibeqc_resident_post_run`` the action runs after a
*successful* evaluation.
"""

from vibeqc_compiler.tensor.cuda_dtype import scalar_type, symmetry_tolerance
from vibeqc_compiler.tensor.cuda_emit import _launch as _emit_launch
from vibeqc_compiler.tensor.cuda_emit import emit_cuda


def _flat_parts(permuted_axes, shape):
    """Row-major flat index of a symmetry partner — same arithmetic as the
    ordinary interpreter's C-order symmetry verification."""
    strides = []
    for axis in range(len(shape)):
        tail = 1
        for extent in shape[axis + 1 :]:
            tail *= extent
        strides.append(tail)
    terms = []
    for axis in range(len(shape)):
        terms.append(
            f"((z) / {max(1, strides[axis])}LL % {max(1, shape[axis])}LL)"
            f" * {max(1, strides[permuted_axes[axis]])}LL"
        )
    return " + ".join(terms)


def _validation_body(plan):
    """Emit one finiteness/symmetry validation kernel per input slot, matching
    ``PreparedCuda._validate`` tolerances exactly."""
    validations, calls = [], []
    for slot, i in enumerate(plan.inputs):
        step = plan.steps[i]
        node = step.node
        ty = scalar_type(node.spec.dtype).ctype
        atol, rtol = symmetry_tolerance(node.spec.dtype)
        comparisons = []
        for symmetry in node.spec.symmetries:
            partner = _flat_parts(symmetry.permutation, node.spec.shape)
            comparisons.append(
                f"double peer = values[{partner}]; if (!isfinite(peer) || "
                f"fabs(value - ({symmetry.sign}) * peer) > {atol} + "
                f"{rtol} * fabs(peer)) atomicCAS(error, 0, {i + 1});"
            )
        body = "".join(f"{{{c}}}" for c in comparisons)
        validations.append(f"""
__global__ void resident_validate_{slot}(unsigned char* p, int* error) {{
    auto* values = reinterpret_cast<const {ty}*>(p + {step.offset});
    for (int z = int(blockIdx.x) * blockDim.x + threadIdx.x; z < {node.spec.size};
         z += int(blockDim.x) * gridDim.x) {{
        double value = values[z];
        if (!isfinite(value)) atomicCAS(error, 0, {i + 1});
        {body}
    }}
}}
""")
        if node.spec.size:
            calls.append(
                f"resident_validate_{slot}<<<vibeqc_tensor::blocks({node.spec.size}, 128),"
                f"128,0,ctx.stream>>>(p,ctx.error); cuda_check(cudaGetLastError());"
            )
    return "".join(validations), calls


def resident_source(plan, *, prefix="", extension=""):
    """Append the resident ABI to the verified ordinary TU.

    ``prefix`` must match the value used to compile the ordinary artifact.
    ``extension`` supplies optional plan-specific tables/kernels.  If it
    defines ``vibeqc_resident_post_run``, that action runs after every
    *successful* evaluation.
    """
    for _, i in plan.outputs:
        step = plan.steps[i]
        if step.virtual or step.last_use != len(plan.steps):
            raise ValueError(
                "resident outputs must be materialized and pinned"
                " for the full plan lifetime"
            )
    for i in plan.inputs:
        step = plan.steps[i]
        if step.virtual or step.last_use != len(plan.steps):
            raise ValueError(
                "resident inputs must be pinned for the full plan lifetime"
            )
    base = emit_cuda(plan, symbol_prefix=prefix)
    validations, _vc = _validation_body(plan)

    inputs = [plan.steps[i] for i in plan.inputs]
    outputs = [plan.steps[i] for _, i in plan.outputs]

    def span_rows(steps):
        return ", ".join(
            f"{{{s.offset}ULL,{s.node.spec.size * s.node.spec.itemsize}ULL}}"
            for s in steps
        )

    # The ordinary ``tensor_run`` performs H2D/D2H copies around the launch
    # sequence.  Resident execution skips those copies by inlining the
    # generated kernels directly, reusing the same section-timer/metrics
    # pattern.  Every contraction, gemm call, offset and alignment is
    # identical — only the per-call staging is absent.
    launches = "".join(_emit_launch(plan, i, prefix) for i in range(len(plan.steps)))
    validation_block = ""
    if _vc:
        validation_block = (
            "ctx.section(profile, metrics.input_ms, [&] {\n"
            + "".join(f"    {call}\n" for call in _vc)
            + "});"
        )

    if extension and "__VIBEQC_RESIDENT_POST_RUN_DECL__" in extension:
        post_run = (
            "int status = vibeqc_resident_post_run(pointer, profile, result, error, size);"
            "\n    if (status) return status;"
        )
    else:
        post_run = ""

    return f"""{base}

#include "cuda_resident.cuh"
{validations}
namespace vibeqc_resident {{
static const Span resident_inputs[] = {{ {span_rows(inputs)} }};
static const Span resident_outputs[] = {{ {span_rows(outputs)} }};
}}  // namespace vibeqc_resident

{extension}
extern "C" const char* resident_plan_identity() {{
    return "{plan.identity}:resident-abi-v1";
}}
extern "C" int resident_abi() {{ return 1; }}
extern "C" int resident_upload(void* pointer, size_t slot, const void* host, size_t bytes,
                               char* error, size_t size) {{
    return vibeqc_resident::transfer(pointer, vibeqc_resident::resident_inputs,
                                     {len(inputs)}, slot, const_cast<void*>(host), bytes,
                                     true, error, size);
}}
extern "C" int resident_run(void* pointer, int profile, Metrics* result, char* error,
                            size_t size) {{
    if (!pointer || !result) {{
        vibeqc_tensor::error_text(error, size, "null resident argument"); return 1;
    }}
    auto& ctx = *static_cast<Context*>(pointer);
    std::unique_lock<std::mutex> lock(ctx.mutex, std::try_to_lock);
    if (!lock.owns_lock()) {{
        vibeqc_tensor::error_text(error, size, "resident plan is busy"); return 1;
    }}
    try {{
        ctx.check_device();
        Metrics metrics;
        metrics.owned_device_bytes = ctx.metrics.owned_device_bytes;
        metrics.provider_retained_bytes = ctx.metrics.provider_retained_bytes;
        metrics.prepare_device_delta = ctx.metrics.prepare_device_delta;
        auto* p = ctx.arena;
        cuda_check(cudaEventRecord(ctx.begin, ctx.stream));
        cuda_check(cudaMemsetAsync(ctx.error, 0, sizeof(int), ctx.stream));
        {validation_block}
        {launches}
        int arithmetic_error = 0;
        cuda_check(cudaMemcpyAsync(&arithmetic_error, ctx.error, sizeof(int),
                                   cudaMemcpyDeviceToHost, ctx.stream));
        cuda_check(cudaEventRecord(ctx.end, ctx.stream));
        cuda_check(cudaEventSynchronize(ctx.end));
        float elapsed = 0;
        cuda_check(cudaEventElapsedTime(&elapsed, ctx.begin, ctx.end));
        metrics.device_ms = elapsed;
        metrics.observed_device_delta = std::max(ctx.metrics.prepare_device_delta,
                                                  ctx.device_delta());
        *result = metrics;
        if (arithmetic_error)
            throw std::runtime_error(std::string(
                arithmetic_error < 0 ? "tensor division by zero at step "
                                     : "non-finite tensor at step ")
                + std::to_string(std::abs(arithmetic_error) - 1));
        (void)ctx;
        {post_run}
        return 0;
    }} catch (const std::exception& e) {{
        cudaStreamSynchronize(ctx.stream);
        vibeqc_tensor::error_text(error, size, e.what()); return 1;
    }}
}}
extern "C" int resident_download(void* pointer, size_t slot, void* host, size_t bytes,
                                 char* error, size_t size) {{
    return vibeqc_resident::transfer(pointer, vibeqc_resident::resident_outputs,
                                     {len(outputs)}, slot, host, bytes,
                                     false, error, size);
}}
"""
