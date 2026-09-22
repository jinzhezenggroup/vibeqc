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
    lowering_diagnostics,
)
from vibeqc_compiler.common.provenance import canonical_hash

from .cuda_gemm import gemm_contract
from .cuda_plan import TensorPlan

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
            shape = tuple(node.spec.shape)
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
        if uses_cublas:
            providers = (CUBLAS_PROVIDER, GENERATED_CUDA_PROVIDER)
            implementation = f"tensor-gemm-{step.gemm}"
        elif is_gemm:
            providers = (CUDA_RUNTIME_PROVIDER,)
            implementation = "tensor-gemm-zero-fill"
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


def tensor_lowering_diagnostics(plan: TensorPlan) -> dict[str, object]:
    """Return deterministic provider provenance for a resolved Tensor CUDA plan."""

    return lowering_diagnostics(resolved_lowering_candidates(plan))
