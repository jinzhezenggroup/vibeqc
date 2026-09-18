"""Shared CUDA benchmark execution-profile resolution."""

from pathlib import Path

import pytest
from vibeqc_compiler.common.cuda_adapter import (
    CudaBenchmarkExecutor,
    CudaExecutionProfile,
    resolve_cuda_execution_profile,
)


def test_cuda_execution_profile_preserves_current_cluster_default():
    profile = resolve_cuda_execution_profile(environment={})
    assert profile == CudaExecutionProfile(
        local=False,
        srun="srun",
        partition="main",
        gres="gpu:5090:1",
        nodes=1,
        ntasks=1,
        slurm_time="00:10:00",
    )
    assert profile.wrap(["python", "worker.py"]) == [
        "srun",
        "--partition=main",
        "--gres=gpu:5090:1",
        "--nodes=1",
        "--ntasks=1",
        "--time=00:10:00",
        "python",
        "worker.py",
    ]


def test_cuda_execution_profile_environment_overrides_project_defaults():
    profile = resolve_cuda_execution_profile(
        environment={
            "VIBEQC_BENCHMARK_PARTITION": "accelerated",
            "VIBEQC_BENCHMARK_GRES": "gpu:a100:2",
            "VIBEQC_BENCHMARK_NODES": "2",
            "VIBEQC_BENCHMARK_NTASKS": "4",
            "VIBEQC_BENCHMARK_TIME": "00:25:00",
            "VIBEQC_BENCHMARK_SRUN": "/opt/slurm/bin/srun",
        }
    )
    assert profile.partition == "accelerated"
    assert profile.gres == "gpu:a100:2"
    assert profile.nodes == 2
    assert profile.ntasks == 4
    assert profile.slurm_time == "00:25:00"
    assert profile.srun == "/opt/slurm/bin/srun"


def test_explicit_execution_arguments_override_environment():
    profile = resolve_cuda_execution_profile(
        environment={
            "VIBEQC_BENCHMARK_PARTITION": "environment",
            "VIBEQC_BENCHMARK_GRES": "gpu:environment:1",
            "VIBEQC_BENCHMARK_NODES": "9",
            "VIBEQC_BENCHMARK_TIME": "01:00:00",
            "VIBEQC_BENCHMARK_LOCAL": "0",
        },
        local=True,
        partition="argument",
        gres="gpu:argument:3",
        nodes=3,
        slurm_time="00:03:00",
    )
    assert profile.local is True
    assert profile.partition == "argument"
    assert profile.gres == "gpu:argument:3"
    assert profile.nodes == 3
    assert profile.slurm_time == "00:03:00"
    assert profile.wrap(["worker"]) == ["worker"]


def test_empty_environment_scheduler_fields_disable_optional_flags():
    profile = resolve_cuda_execution_profile(
        environment={
            "VIBEQC_BENCHMARK_PARTITION": "",
            "VIBEQC_BENCHMARK_GRES": "",
            "VIBEQC_BENCHMARK_TIME": "",
        }
    )
    assert profile.wrap(["worker"]) == [
        "srun",
        "--nodes=1",
        "--ntasks=1",
        "worker",
    ]


def test_executor_uses_caller_time_as_default_but_environment_can_override():
    executor = CudaBenchmarkExecutor.from_environment(
        30,
        default_slurm_time="00:30:00",
        environment={"VIBEQC_BENCHMARK_TIME": "00:04:00"},
    )
    assert executor.profile.slurm_time == "00:04:00"
    assert executor.command(Path("benchmark"))[-2:] == ["--time=00:04:00", "benchmark"]


@pytest.mark.parametrize(
    ("environment", "match"),
    [
        ({"VIBEQC_BENCHMARK_LOCAL": "sometimes"}, "boolean"),
        ({"VIBEQC_BENCHMARK_NODES": "many"}, "integer"),
    ],
)
def test_invalid_execution_environment_fails_closed(environment, match):
    with pytest.raises(ValueError, match=match):
        resolve_cuda_execution_profile(environment=environment)
