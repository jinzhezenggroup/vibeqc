"""CPU allocation is explicit without breaking existing positional profiles."""

from pathlib import Path

import pytest
from vibeqc_compiler.common.cuda_adapter import (
    CudaBenchmarkExecutor,
    CudaExecutionProfile,
    resolve_cuda_execution_profile,
)


@pytest.mark.parametrize("cpus", (True, False, 1.5, float("nan"), "4", 0, -1))
@pytest.mark.parametrize("path", ("profile", "resolver", "executor"))
def test_cpu_allocation_rejects_nonpositive_or_noninteger_counts(
    cpus: object, path: str
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        if path == "profile":
            CudaExecutionProfile(cpus_per_task=cpus)
        elif path == "resolver":
            resolve_cuda_execution_profile(environment={}, cpus_per_task=cpus)
        else:
            CudaBenchmarkExecutor(30, cpus_per_task=cpus).command(Path("worker"))


def test_cpu_allocation_keeps_legacy_positional_time() -> None:
    profile = CudaExecutionProfile(False, "srun", "main", "gpu:1", 1, 1, "00:03:00")
    executor = CudaBenchmarkExecutor(
        30, False, "srun", "main", "gpu:1", 1, 1, "00:03:00"
    )
    assert profile.slurm_time == executor.profile.slurm_time == "00:03:00"
    assert profile.cpus_per_task is None and executor.profile.cpus_per_task is None
    assert "--time=00:03:00" in profile.wrap(["worker"])


def test_cpu_allocation_single_task_launch_and_serial_default() -> None:
    profile = resolve_cuda_execution_profile(
        environment={"VIBEQC_BENCHMARK_CPUS_PER_TASK": "8"}
    )
    assert profile.wrap(["worker"]).count("--cpus-per-task=8") == 1
    assert profile.to_dict()["cpus_per_task"] == 8
    assert not any(
        flag.startswith("--cpus-per-task")
        for flag in resolve_cuda_execution_profile(environment={}).wrap(["worker"])
    )
