"""Shared ScheduleIR contract adapters and cross-consumer diagnostics."""

from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.schedule import ScheduleContract, schedule_diagnostics
from vibeqc_compiler.dft.xc_schedule import (
    DEVICE_FUSED,
    GridXcCandidateLimits,
    GridXcCandidateShape,
    GridXcScientificIdentity,
    assess_grid_xc_schedule,
    schedule_profile_key,
)
from vibeqc_compiler.integral.one_electron_derivative_policy_cuda import (
    one_electron_derivative_schedule_contract,
)
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    input_tensor,
)
from vibeqc_compiler.tensor.cuda_plan import plan_cuda
from vibeqc_compiler.tensor.cuda_search import estimate_schedule


def _tensor_contract() -> ScheduleContract:
    index = Index("i", IndexSpace("axis", "batch", 17))
    value = input_tensor("x", TensorSpec((index,), role="input"))
    plan = plan_cuda(
        Program({"result": add(value, value)}),
        cuda_target_info("sm_80"),
    )
    return ScheduleContract.from_payload(estimate_schedule(plan)["schedule_contract"])


def _dft_contract() -> ScheduleContract:
    scientific = GridXcScientificIdentity(
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
    assessment = assess_grid_xc_schedule(
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
        scientific=scientific,
    )
    assert assessment.schedule_contract.profile_key == schedule_profile_key(scientific)
    return assessment.schedule_contract


def test_tensor_dft_and_integral_use_one_schedule_contract_vocabulary() -> None:
    tensor = _tensor_contract()
    dft = _dft_contract()
    integral = one_electron_derivative_schedule_contract(
        cuda_target_info("sm_120").target_info
    )
    report = schedule_diagnostics((tensor, dft, integral))

    assert tensor.consumer == "tensor.cuda"
    assert dft.consumer == "dft.grid_xc"
    assert integral.consumer == "integral.one_electron_derivative"
    assert tensor.topology.workgroup_threads == 128
    assert dft.topology.tiles == (256,)
    assert tensor.workload_hash is not None and dft.workload_hash is not None
    assert tensor.precision_schedule_hash is not None
    assert dft.precision_schedule_hash is not None
    assert integral.profile_key is not None
    assert integral.topology.cooperative
    assert set(report["consumers"]) == {
        "tensor.cuda",
        "dft.grid_xc",
        "integral.one_electron_derivative",
    }
    assert set(report["static_order"]) == {
        tensor.identity,
        dft.identity,
        integral.identity,
    }
    assert set(report["contracts"][0]["resources"]) == set(
        report["contracts"][1]["resources"]
    )


def test_shared_contract_roundtrip_preserves_owner_schedule_identity() -> None:
    for contract in (
        _tensor_contract(),
        _dft_contract(),
        one_electron_derivative_schedule_contract(
            cuda_target_info("sm_120").target_info
        ),
    ):
        replay = ScheduleContract.from_payload(contract.to_payload())
        assert replay == contract
        assert replay.identity == contract.identity
        assert replay.schedule_hash == contract.schedule_hash
