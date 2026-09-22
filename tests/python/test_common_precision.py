"""Cross-IR execution-precision contracts for TensorIR, DFT, and integrals."""

import pytest
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.precision import (
    ExecutionPrecisionSchedule,
    PrecisionDirective,
    uniform_precision_schedule,
)
from vibeqc_compiler.dft.xc_schedule import (
    DEVICE_FUSED,
    GridXcCandidateAssessment,
    GridXcCandidateLimits,
    GridXcCandidateShape,
    GridXcScientificIdentity,
    assess_grid_xc_schedule,
)
from vibeqc_compiler.integral import generated_fock_precision_schedule
from vibeqc_compiler.integral.one_electron_derivative_policy_cuda import (
    one_electron_derivative_schedule_contract,
)
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    describe_precision,
    input_tensor,
)
from vibeqc_compiler.tensor import (
    PrecisionDirective as TensorPrecisionDirective,
)


def _dft_scientific() -> GridXcScientificIdentity:
    return GridXcScientificIdentity(
        architecture="sm_120",
        functional="PBE",
        functional_identity="f" * 64,
        ingredients=("rho", "gradient", "sigma"),
        jet_outputs=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
        grid_identity="g" * 64,
        grid_model="GridSpec-v2",
        screening_identity="m" * 64,
        precision="fp64",
        spin="polarized",
        observable="potential",
        density_route="density_matrix",
        source_identity="s" * 64,
    )


def _dft_assessment(
    precision: ExecutionPrecisionSchedule | None = None,
) -> GridXcCandidateAssessment:
    return assess_grid_xc_schedule(
        DEVICE_FUSED,
        GridXcCandidateShape(
            npoint=2048,
            tile_points=256,
            nao=64,
            max_active_ao=32,
            spins=2,
            jet_components=4,
            device_workspace_bytes=4 << 20,
            generated_source_bytes=120_000,
        ),
        GridXcCandidateLimits(
            device_bytes=16 << 20,
            live_values=1_000_000,
            source_bytes=300_000,
        ),
        device_xc_available=True,
        observable="potential",
        functional="PBE",
        scientific=_dft_scientific(),
        precision_schedule=precision,
    )


def test_tensor_precision_reuses_common_directive_and_projects_schedule() -> None:
    assert TensorPrecisionDirective is PrecisionDirective
    index = Index("i", IndexSpace("axis", "batch", 17))
    value = input_tensor("x", TensorSpec((index,), role="input"))
    schedule = describe_precision(Program({"result": add(value, value)}))
    shared = schedule.execution_precision
    assert shared.is_strict_fp64
    assert shared.strict_audit_dtype == "float64"
    assert shared.audit_owner == "method-controller"
    assert shared.regions


def test_execution_precision_identity_is_region_order_independent() -> None:
    fp64 = PrecisionDirective("float64", "float64", "float64")
    first = ExecutionPrecisionSchedule((("b", fp64), ("a", fp64)))
    second = ExecutionPrecisionSchedule((("a", fp64), ("b", fp64)))
    assert first.regions == second.regions
    assert first.identity == second.identity
    with pytest.raises(ValueError, match="duplicate precision region"):
        ExecutionPrecisionSchedule((("a", fp64), ("a", fp64)))


def test_dft_uses_common_precision_identity_and_fails_closed_on_mixed() -> None:
    strict = uniform_precision_schedule("dft.grid_xc")
    accepted = _dft_assessment(strict)
    assert accepted.legal
    assert accepted.schedule_contract.precision_schedule_hash == strict.identity
    assert accepted.schedule_contract.profitability.precision_cast_bytes == 0
    assert (
        accepted.schedule_contract.profitability.precision_widened_accumulation_terms
        == 0
    )

    mixed = uniform_precision_schedule(
        "dft.grid_xc",
        storage_dtype="float32",
        compute_dtype="float32",
        accumulation_dtype="float64",
        qualification="test-only",
    )
    rejected = _dft_assessment(mixed)
    assert not rejected.legal
    assert rejected.schedule_contract.precision_schedule_hash == mixed.identity
    assert (
        "grid/XC lowering currently supports only strict FP64 execution precision"
        in rejected.reasons
    )


def test_integral_schedule_contract_uses_common_precision_identity() -> None:
    contract = one_electron_derivative_schedule_contract(
        cuda_target_info("sm_120").target_info
    )
    assert (
        contract.precision_schedule_hash
        == uniform_precision_schedule("integral.one_electron_derivative").identity
    )
    assert contract.profitability.precision_cast_bytes == 0
    assert contract.profitability.precision_widened_accumulation_terms == 0


def test_generated_fock_mixed_schedule_records_fp32_eri_fp64_accumulation() -> None:
    strict = generated_fock_precision_schedule()
    assert strict.is_strict_fp64

    with pytest.raises(ValueError, match="requires a qualification"):
        generated_fock_precision_schedule(mixed_eri=True)

    mixed = generated_fock_precision_schedule(
        mixed_eri=True,
        qualification="scf-mixed-fock-qualified-domain",
    )
    regions = dict(mixed.regions)
    assert regions["eri_recurrence"].compute_dtype == "float32"
    assert regions["eri_recurrence"].accumulation_dtype == "float32"
    assert regions["fock_accumulation"].storage_dtype == "float64"
    assert regions["fock_accumulation"].accumulation_dtype == "float64"
    assert mixed.strict_audit_dtype == "float64"
    assert not mixed.is_strict_fp64


@pytest.mark.parametrize("candidate_local", (False, True))
@pytest.mark.parametrize("mixed", (False, True))
def test_grid_ranking_preserves_explicit_precision_after_candidate_refactor(
    candidate_local: bool, mixed: bool
) -> None:
    from vibeqc_compiler.dft.xc_schedule import (
        GridXcScheduleCandidate,
        rank_grid_xc_candidates,
        rank_grid_xc_schedules,
    )

    shape = GridXcCandidateShape(
        npoint=256,
        tile_points=128,
        nao=8,
        max_active_ao=8,
        spins=2,
        jet_components=4,
        device_workspace_bytes=4096,
        generated_source_bytes=4096,
    )
    limits = GridXcCandidateLimits(
        device_bytes=1 << 20,
        live_values=1_000_000,
        source_bytes=1 << 20,
    )
    precision = uniform_precision_schedule(
        "explicit-grid-review",
        compute_dtype="float32" if mixed else "float64",
        qualification="review-regression",
    )
    kwargs = {
        "device_xc_available": True,
        "observable": "potential",
        "functional": "PBE",
        "precision_schedule": precision,
    }
    if candidate_local:
        ranked = rank_grid_xc_candidates(
            (GridXcScheduleCandidate(DEVICE_FUSED, shape),), limits, **kwargs
        )
    else:
        ranked = rank_grid_xc_schedules((DEVICE_FUSED,), shape, limits, **kwargs)
    if mixed:
        assert ranked == ()
    else:
        assert len(ranked) == 1
        assert ranked[0].schedule_contract.precision_schedule_hash == precision.identity
