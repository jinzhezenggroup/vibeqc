"""Resolved provider provenance for the existing TensorIR CUDA lowering.

This module does not choose a new implementation.  It projects the already
planned TensorIR execution into the shared lowering-provider contract so future
cuBLASLt/CUTLASS/CUB candidates can be compared without adding provider-specific
branches to scientific IR.
"""

from __future__ import annotations

from dataclasses import asdict

from vibeqc_compiler.common.lowering_provider import (
    LoweringCandidate,
    LoweringRequest,
    ProviderDescriptor,
    collect_lowering_candidates,
    lowering_diagnostics,
)
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.specialization import TargetCapabilities

from .cuda_gemm import gemm_contract
from .cuda_plan import TensorPlan
from .cuda_reduction import cooperative_reduction_provider, reduction_extent

GENERATED_CUDA_PROVIDER = ProviderDescriptor(
    name="vibeqc.generated_cuda",
    kind="generated",
    implementation="tensor-cuda-emitter",
    provenance=(("version_source", "compiler-source-identity"),),
)

CUBLAS_PROVIDER = ProviderDescriptor(
    name="nvidia.cublas",
    kind="library",
    implementation="cublas-gemm",
    required_features=("cublas",),
    provenance=(("version_source", "runtime-probe"),),
)


CUDA_RUNTIME_PROVIDER = ProviderDescriptor(
    name="nvidia.cuda_runtime",
    kind="runtime",
    implementation="cuda-runtime",
    required_features=("cuda-runtime",),
    provenance=(("version_source", "runtime-probe"),),
)


CUB_REDUCTION_PROVIDER = ProviderDescriptor(
    name="nvidia.cccl.cub",
    kind="library",
    implementation="cub-block-reduce",
    required_features=("cuda", "cub-block-reduce-header"),
    provenance=(("version_source", "cuda-toolkit-cccl-header"),),
)


class GeneratedReductionProvider:
    """Advertise the existing generated cooperative reduction."""

    descriptor = GENERATED_CUDA_PROVIDER

    def candidates(
        self, request: LoweringRequest, target: TargetCapabilities
    ) -> tuple[LoweringCandidate, ...]:
        reason = _cooperative_reduction_rejection(request, target)
        return (
            LoweringCandidate(
                request=request,
                implementation="tensor-reduce-generated-cooperative",
                providers=(self.descriptor,),
                status="unsupported" if reason else "ready",
                numerical_mode=f"{request.dtype}->{request.accumulation_dtype}",
                reason=reason,
            ),
        )


class CubReductionProvider:
    """Advertise the opt-in CUB BlockReduce implementation."""

    descriptor = CUB_REDUCTION_PROVIDER

    def candidates(
        self, request: LoweringRequest, target: TargetCapabilities
    ) -> tuple[LoweringCandidate, ...]:
        reason = _cooperative_reduction_rejection(request, target)
        if (
            reason is None
            and dict(target.features).get("cub-block-reduce-header") is not True
        ):
            reason = "CUB requires explicit cub-block-reduce-header capability"
        return (
            LoweringCandidate(
                request=request,
                implementation="tensor-reduce-cub-block-reduce",
                providers=(self.descriptor, GENERATED_CUDA_PROVIDER),
                status="unsupported" if reason else "ready",
                numerical_mode=f"{request.dtype}->{request.accumulation_dtype}",
                reason=reason,
            ),
        )


def _cooperative_reduction_rejection(
    request: LoweringRequest, target: TargetCapabilities
) -> str | None:
    if request.operation != "reduce" or len(request.shape) != 2:
        return "provider requires a flattened TensorIR reduction request"
    if request.dtype not in (
        "float32",
        "float64",
    ) or request.accumulation_dtype not in (
        "float32",
        "float64",
    ):
        return "provider supports float32/float64 reduction arithmetic only"
    subgroup = target.target.subgroup_size
    if subgroup != 32:
        return "cooperative reduction pilot requires CUDA subgroup size 32"
    if request.shape[1] < subgroup:
        return "reduction extent is below the cooperative subgroup threshold"
    return None


def _site_hash(plan: TensorPlan, index: int) -> str:
    step = plan.steps[index]
    contract = gemm_contract(step.node)
    payload: dict[str, object] = {
        "program": plan.program.logical_hash,
        "step": index,
        "operation": step.node.op,
        "dtype": step.node.spec.dtype,
        "shape": step.node.spec.shape,
    }
    if contract is not None:
        payload["gemm"] = asdict(contract)
    return canonical_hash(payload)


def _numerical_mode(plan: TensorPlan, index: int) -> str:
    node = plan.steps[index].node
    value = plan.precision_by_node.get(node)
    if value is None:
        return f"{node.spec.dtype}->{node.spec.dtype}"
    return f"{value.compute_dtype}->{value.accumulation_dtype}"


def resolved_lowering_candidates(plan: TensorPlan) -> tuple[LoweringCandidate, ...]:
    """Describe the providers already selected by one TensorPlan.

    GEMM is a composite lowering: cuBLAS owns the contraction, while generated
    CUDA owns packing/scatter and/or the checked coefficient epilogue.  Empty-K
    contractions use the CUDA runtime zero-fill path and therefore do not claim
    a generated kernel or cuBLAS provider.
    """

    if not isinstance(plan, TensorPlan):
        raise TypeError("Tensor CUDA lowering diagnostics require a TensorPlan")
    candidates: list[LoweringCandidate] = []
    for index, step in enumerate(plan.steps):
        node = step.node
        if step.virtual or node.op in ("input", "constant") or not node.spec.size:
            continue
        contract = gemm_contract(node)
        if step.gemm != "none" and contract is not None:
            is_gemm = True
            uses_cublas = contract.k > 0
            shape = (contract.batch, contract.m, contract.n, contract.k)
        else:
            is_gemm = False
            uses_cublas = False
            shape = (
                (node.spec.size, reduction_extent(node))
                if node.op == "reduce"
                else tuple(node.spec.shape)
            )
        value_precision = plan.precision_by_node.get(node)
        request = LoweringRequest(
            consumer="tensor.cuda",
            operation="gemm" if is_gemm else node.op,
            backend="cuda",
            dtype=node.spec.dtype,
            accumulation_dtype=(
                value_precision.accumulation_dtype
                if value_precision is not None
                else node.spec.dtype
            ),
            shape=shape,
            semantics=(
                ("program_hash", plan.program.logical_hash),
                ("site_hash", _site_hash(plan, index)),
            ),
        )
        reduction_provider = cooperative_reduction_provider(plan, index)
        if uses_cublas:
            providers = (CUBLAS_PROVIDER, GENERATED_CUDA_PROVIDER)
            implementation = f"tensor-gemm-{step.gemm}"
        elif is_gemm:
            providers = (CUDA_RUNTIME_PROVIDER,)
            implementation = "tensor-gemm-zero-fill"
        elif reduction_provider == "cub":
            providers = (CUB_REDUCTION_PROVIDER, GENERATED_CUDA_PROVIDER)
            implementation = "tensor-reduce-cub-block-reduce"
        elif reduction_provider == "generated":
            providers = (GENERATED_CUDA_PROVIDER,)
            implementation = "tensor-reduce-generated-cooperative"
        else:
            providers = (GENERATED_CUDA_PROVIDER,)
            implementation = f"tensor-generated-{node.op}"
        candidates.append(
            LoweringCandidate(
                request=request,
                implementation=implementation,
                providers=providers,
                status="ready",
                numerical_mode=_numerical_mode(plan, index),
                workspace_bytes=plan.library_bytes if uses_cublas else 0,
                provider_bytes=plan.provider_bytes if uses_cublas else 0,
                provenance=(
                    ("plan_identity", plan.identity),
                    ("step_index", index),
                ),
            )
        )
    return tuple(candidates)


def reduction_provider_candidates(
    plan: TensorPlan,
    index: int,
    *,
    target_capabilities: TargetCapabilities | None = None,
) -> tuple[LoweringCandidate, ...]:
    """Advertise reductions using explicit, plan-bound toolkit capability facts.

    GPU architecture alone does not establish that CUB headers are installed.
    Without caller-supplied header evidence the CUB offer is unsupported; the
    generated offer remains available. This routine does not probe a toolkit.
    """

    if not isinstance(plan, TensorPlan):
        raise TypeError("reduction provider candidates require a TensorPlan")
    if cooperative_reduction_provider(plan, index) is None:
        raise ValueError("step is not an eligible cooperative reduction")
    step = plan.steps[index]
    node = step.node
    value_precision = plan.precision_by_node[node]
    request = LoweringRequest(
        consumer="tensor.cuda",
        operation="reduce",
        backend="cuda",
        dtype=node.spec.dtype,
        accumulation_dtype=value_precision.accumulation_dtype,
        shape=(node.spec.size, reduction_extent(node)),
        semantics=(
            ("program_hash", plan.program.logical_hash),
            ("site_hash", _site_hash(plan, index)),
        ),
    )
    target = TargetCapabilities(
        plan.target.target_info,
        features=tuple(
            (feature, True) for feature in plan.target.required_cuda_features
        ),
    )
    if target_capabilities is not None:
        if not isinstance(target_capabilities, TargetCapabilities):
            raise TypeError("reduction capabilities require TargetCapabilities")
        if target_capabilities.target != plan.target.target_info:
            raise ValueError("reduction capabilities do not match the planned target")
        target = target_capabilities
    return collect_lowering_candidates(
        request,
        target,
        (GeneratedReductionProvider(), CubReductionProvider()),
    )


def tensor_lowering_diagnostics(plan: TensorPlan) -> dict[str, object]:
    """Return deterministic provider provenance for a resolved Tensor CUDA plan."""

    return lowering_diagnostics(resolved_lowering_candidates(plan))
