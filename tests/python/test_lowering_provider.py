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
    reduce_sum,
)
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda
from vibeqc_compiler.tensor.cuda_providers import (
    CubReductionProvider,
    reduction_provider_candidates,
    tensor_lowering_diagnostics,
)
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


def _reduction_program() -> Program:
    i = Index("i", IndexSpace("rows", "batch", 17))
    k = Index("k", IndexSpace("inner", "batch", 129))
    x = input_tensor("x", TensorSpec((i, k), role="input"))
    return Program({"result": reduce_sum(x, (1,))})


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


def test_generated_and_cub_reduction_providers_share_one_request() -> None:
    program = _reduction_program()
    generated = plan_cuda(
        program,
        TARGET,
        schedule=TensorSchedule(stream_reductions=True),
    )
    index = next(
        i for i, step in enumerate(generated.steps) if step.node.op == "reduce"
    )
    unknown = reduction_provider_candidates(generated, index)
    assert [candidate.status for candidate in unknown] == ["ready", "unsupported"]
    capabilities = TargetCapabilities(
        TARGET.target_info, features=(("cub-block-reduce-header", True),)
    )
    candidates = reduction_provider_candidates(
        generated, index, target_capabilities=capabilities
    )
    foreign = TargetCapabilities(
        cuda_target_info("sm_120").target_info,
        features=(("cub-block-reduce-header", True),),
    )
    with pytest.raises(ValueError, match="planned target"):
        reduction_provider_candidates(generated, index, target_capabilities=foreign)

    assert [candidate.status for candidate in candidates] == ["ready", "ready"]
    assert candidates[0].request == candidates[1].request
    assert candidates[0].implementation == "tensor-reduce-generated-cooperative"
    assert candidates[1].implementation == "tensor-reduce-cub-block-reduce"
    assert [provider.name for provider in candidates[1].providers] == [
        "nvidia.cccl.cub",
        "vibeqc.generated_cuda",
    ]

    cub = plan_cuda(
        program,
        TARGET,
        schedule=TensorSchedule(
            stream_reductions=True,
            reduction_provider="cub",
        ),
    )
    report = tensor_lowering_diagnostics(cub)
    selected = next(
        row
        for row in report["candidates"]
        if row["implementation"] == "tensor-reduce-cub-block-reduce"
    )
    assert [provider["name"] for provider in selected["providers"]] == [
        "nvidia.cccl.cub",
        "vibeqc.generated_cuda",
    ]


def test_schedule_contract_carries_resolved_lowering_identity() -> None:
    plan = plan_cuda(_gemm_program(), TARGET)
    lowering = tensor_lowering_diagnostics(plan)
    contract = ScheduleContract.from_payload(
        estimate_schedule(plan)["schedule_contract"]
    )
    provenance = dict(contract.provenance)

    assert provenance["lowering_identity"] == lowering["identity"]
    assert provenance["lowering_providers"] == "nvidia.cublas,vibeqc.generated_cuda"


@pytest.mark.parametrize("evidence", [None, False, 0, 1, "true"])
def test_cub_provider_requires_explicit_typed_header_evidence(evidence: object) -> None:
    request = LoweringRequest(
        consumer="tensor.cuda",
        operation="reduce",
        backend="cuda",
        dtype="float64",
        accumulation_dtype="float64",
        shape=(17, 129),
    )
    features = () if evidence is None else (("cub-block-reduce-header", evidence),)
    target = TargetCapabilities(TARGET.target_info, features=features)
    offered = CubReductionProvider().candidates(request, target)
    assert len(offered) == 1
    assert offered[0].status == "unsupported"
    assert "cub-block-reduce-header" in offered[0].reason


def test_cub_provider_accepts_explicit_header_evidence() -> None:
    request = LoweringRequest(
        consumer="tensor.cuda",
        operation="reduce",
        backend="cuda",
        dtype="float64",
        accumulation_dtype="float64",
        shape=(17, 129),
    )
    target = TargetCapabilities(
        TARGET.target_info, features=(("cub-block-reduce-header", True),)
    )
    assert CubReductionProvider().candidates(request, target)[0].status == "ready"
