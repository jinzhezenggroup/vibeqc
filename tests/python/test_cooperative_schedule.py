"""Shared cooperative lane ownership across generated derivative consumers."""

import re

from vibeqc_compiler.common.backend import TargetInfo
from vibeqc_compiler.common.cuda_target import CUDA_TARGETS
from vibeqc_compiler.common.schedule import ScheduleContract
from vibeqc_compiler.integral.cooperative_schedule import CooperativeLaneSchedule
from vibeqc_compiler.integral.df_rys_shell import COOPERATIVE_RYS_SHELL_CLASSES
from vibeqc_compiler.integral.df_shell_derivatives import (
    emit_df_shell_derivatives_cuda,
    shell_schedule,
)
from vibeqc_compiler.integral.one_electron_derivative_policy_cuda import (
    emit_one_electron_derivative_policy_cuda,
    nucleus_cooperative_schedule,
    one_electron_derivative_policy_inventory,
    production_nucleus_cooperative_schedule,
)
from vibeqc_compiler.integral.one_electron_derivatives_cuda import (
    one_electron_derivative_inventory,
)


def test_nucleus_cooperative_schedule_uses_target_resource_limits() -> None:
    target = TargetInfo(
        backend="cuda",
        architecture="synthetic",
        subgroup_size=32,
        maximum_workgroup_threads=64,
        maximum_resident_workgroups=8,
    )
    schedule = nucleus_cooperative_schedule(target)
    assert schedule == CooperativeLaneSchedule(32, 32, 2)
    assert schedule.workgroup_threads == 64
    schedule.validate_for(target)


def test_production_one_electron_schedule_is_portable_across_cuda_catalog() -> None:
    schedule = production_nucleus_cooperative_schedule()
    assert schedule.lanes_per_group == 32
    assert schedule.groups_per_workgroup == 4
    assert schedule.workgroup_threads == 128
    for target in CUDA_TARGETS.values():
        schedule.validate_for(target.target_info)


def test_df_rys_reuses_shared_cooperative_lane_contract() -> None:
    angular = (1, 1, 1)
    assert angular in COOPERATIVE_RYS_SHELL_CLASSES
    schedule = shell_schedule(angular, 2)
    assert isinstance(schedule.cooperative, CooperativeLaneSchedule)
    assert schedule.component_lanes == schedule.cooperative.lanes_per_group
    assert schedule.triples_per_block == schedule.cooperative.groups_per_workgroup
    schedule.cooperative.validate_for(CUDA_TARGETS["sm_120"].target_info)

    source = emit_df_shell_derivatives_cuda(classes=[angular])
    assert schedule.cooperative.identity in source
    assert "cooperative_schedule_identity" in source


def test_one_electron_policy_emits_shared_schedule_identity_and_geometry() -> None:
    schedule = production_nucleus_cooperative_schedule()
    scientific = one_electron_derivative_inventory()
    policy = one_electron_derivative_policy_inventory()
    assert "schedule" not in scientific
    assert policy["cooperative_schedule_identity"] == schedule.identity
    assert set(policy["schedule_contracts"]) == set(CUDA_TARGETS)
    for architecture, payload in policy["schedule_contracts"].items():
        contract = ScheduleContract.from_payload(payload)
        assert contract.consumer == "integral.one_electron_derivative"
        assert (
            contract.schedule_hash
            == nucleus_cooperative_schedule(
                CUDA_TARGETS[architecture].target_info
            ).identity
        )
        assert contract.profile_key == policy["workload_profile_key"]
        assert contract.topology.cooperative
        assert contract.topology.reduction == "lane-group"
    source = emit_one_electron_derivative_policy_cuda()
    assert schedule.identity in source
    assert "group_lanes=32U" in source
    assert "groups_per_block=4U" in source
    assert "block_threads=128U" in source
    assert re.search(r"schedule_code=3U", source)
