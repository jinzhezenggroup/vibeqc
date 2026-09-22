"""Lowering-provider contracts and TensorIR CUDA provider provenance."""

from __future__ import annotations

import pytest
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.lowering_provider import (
    LoweringCandidate,
    LoweringRequest,
    ProviderDescriptor,
    collect_lowering_candidates,
    lowering_diagnostics,
)
from vibeqc_compiler.common.schedule import ScheduleContract
from vibeqc_compiler.common.specialization import TargetCapabilities
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
)
from vibeqc_compiler.tensor.cuda_plan import plan_cuda
from vibeqc_compiler.tensor.cuda_providers import tensor_lowering_diagnostics
from vibeqc_compiler.tensor.cuda_search import estimate_schedule

TARGET = cuda_target_info("sm_80")


def _vector_program() -> Program:
    i = Index("i", IndexSpace("axis", "batch", 17))
    x = input_tensor("x", TensorSpec((i,), role="input"))
    return Program({"result": add(x, x)})


def _gemm_program(*, packed: bool = False, inner: int = 13) -> Program:
    i = Index("i", IndexSpace("rows", "batch", 7))
    j = Index("j", IndexSpace("cols", "batch", 11))
    k = Index("k", IndexSpace("inner", "batch", inner))
    a = input_tensor("a", TensorSpec((i, k), role="input"))
    b = input_tensor("b", TensorSpec((k, j), role="input"))
    equation = "ik,kj->ji" if packed else "ik,kj->ij"
    return Program({"result": einsum(equation, a, b)})


def test_lowering_contract_is_canonical_and_keeps_negative_evidence() -> None:
    first = LoweringRequest(
        consumer="tensor.cuda",
        operation="gemm",
        backend="cuda",
        dtype="float64",
        accumulation_dtype="float64",
        shape=(7, 11, 13),
        semantics=(("z", 2), ("a", 1)),
    )
    second = LoweringRequest(
        consumer="tensor.cuda",
        operation="gemm",
        backend="cuda",
        dtype="float64",
        accumulation_dtype="float64",
        shape=(7, 11, 13),
        semantics=(("a", 1), ("z", 2)),
    )
    assert first.identity == second.identity

    boolean_semantic = LoweringRequest(
        consumer="tensor.cuda",
        operation="gemm",
        backend="cuda",
        dtype="float64",
        accumulation_dtype="float64",
        shape=(7, 11, 13),
        semantics=(("flag", True),),
    )
    integer_semantic = LoweringRequest(
        consumer="tensor.cuda",
        operation="gemm",
        backend="cuda",
        dtype="float64",
        accumulation_dtype="float64",
        shape=(7, 11, 13),
        semantics=(("flag", 1),),
    )
    assert boolean_semantic != integer_semantic
    assert boolean_semantic.identity != integer_semantic.identity

    provider = ProviderDescriptor(
        name="nvidia.cublaslt",
        kind="library",
        implementation="matmul",
        required_features=("fp64", "cublaslt"),
    )
    rejected = LoweringCandidate(
        request=first,
        implementation="cublaslt-matmul",
        providers=(provider,),
        status="unsupported",
        numerical_mode="float64->float64",
        reason="target toolkit does not expose cuBLASLt",
    )
    report = lowering_diagnostics((rejected,))
    assert report["providers"] == ["nvidia.cublaslt"]
    assert report["candidates"][0]["reason"] == rejected.reason

    with pytest.raises(ValueError, match="rejection reason"):
        LoweringCandidate(
            request=first,
            implementation="invalid",
            providers=(provider,),
            status="unsupported",
            numerical_mode="float64->float64",
        )


class _RejectingProvider:
    def __init__(self, descriptor: ProviderDescriptor) -> None:
        self.descriptor = descriptor

    def candidates(
        self, request: LoweringRequest, target: TargetCapabilities
    ) -> tuple[LoweringCandidate, ...]:
        del target
        return (
            LoweringCandidate(
                request=request,
                implementation="rejected",
                providers=(self.descriptor,),
                status="unsupported",
                numerical_mode=f"{request.dtype}->{request.accumulation_dtype}",
                reason="required target capability is unavailable",
            ),
        )


class _SilentProvider:
    def __init__(self, descriptor: ProviderDescriptor) -> None:
        self.descriptor = descriptor

    def candidates(
        self, request: LoweringRequest, target: TargetCapabilities
    ) -> tuple[LoweringCandidate, ...]:
        del request, target
        return ()


def test_provider_collection_requires_explicit_negative_evidence() -> None:
    request = LoweringRequest(
        consumer="tensor.cuda",
        operation="gemm",
        backend="cuda",
        dtype="float64",
        accumulation_dtype="float64",
        shape=(7, 11, 13),
    )
    descriptor = ProviderDescriptor(
        name="nvidia.cublaslt",
        kind="library",
        implementation="matmul",
        required_features=("cublaslt",),
    )
    target = TargetCapabilities(
        TARGET.target_info,
        features=(("cublaslt", False),),
    )
    candidate = collect_lowering_candidates(
        request, target, (_RejectingProvider(descriptor),)
    )[0]
    assert candidate.status == "unsupported"
    assert candidate.reason == "required target capability is unavailable"

    with pytest.raises(ValueError, match="explicit unsupported evidence"):
        collect_lowering_candidates(request, target, (_SilentProvider(descriptor),))


def test_tensor_generated_cuda_provider_is_explicit() -> None:
    plan = plan_cuda(_vector_program(), TARGET)
    report = tensor_lowering_diagnostics(plan)

    assert report["providers"] == ["vibeqc.generated_cuda"]
    assert report["candidates"]
    assert all(
        candidate["providers"][0]["name"] == "vibeqc.generated_cuda"
        for candidate in report["candidates"]
    )


@pytest.mark.parametrize(
    ("packed", "implementation"),
    [(False, "tensor-gemm-direct-NN"), (True, "tensor-gemm-packed")],
)
def test_tensor_cublas_is_explicit_composite_lowering(
    packed: bool, implementation: str
) -> None:
    plan = plan_cuda(_gemm_program(packed=packed), TARGET)
    report = tensor_lowering_diagnostics(plan)
    candidate = next(
        row for row in report["candidates"] if row["implementation"] == implementation
    )

    assert [provider["name"] for provider in candidate["providers"]] == [
        "nvidia.cublas",
        "vibeqc.generated_cuda",
    ]
    assert candidate["workspace_bytes"] == plan.library_bytes
    assert candidate["provider_bytes"] == plan.provider_bytes


def test_empty_gemm_is_attributed_to_cuda_runtime_zero_fill() -> None:
    plan = plan_cuda(_gemm_program(inner=0), TARGET)
    report = tensor_lowering_diagnostics(plan)
    candidate = next(
        row
        for row in report["candidates"]
        if row["implementation"] == "tensor-gemm-zero-fill"
    )

    assert [provider["name"] for provider in candidate["providers"]] == [
        "nvidia.cuda_runtime"
    ]
    assert candidate["workspace_bytes"] == 0
    assert candidate["provider_bytes"] == 0


def test_schedule_contract_carries_resolved_lowering_identity() -> None:
    plan = plan_cuda(_gemm_program(), TARGET)
    lowering = tensor_lowering_diagnostics(plan)
    contract = ScheduleContract.from_payload(
        estimate_schedule(plan)["schedule_contract"]
    )
    provenance = dict(contract.provenance)

    assert provenance["lowering_identity"] == lowering["identity"]
    assert provenance["lowering_providers"] == "nvidia.cublas,vibeqc.generated_cuda"
