"""Shared CUDA benchmark execution-profile resolution."""

import typing
from pathlib import Path

import pytest
from vibeqc_compiler.common.cuda_adapter import (
    CudaBenchmarkExecutor,
    CudaExecutionProfile,
    resolve_cuda_execution_profile,
)


def test_cuda_execution_profile_preserves_current_cluster_default() -> None:
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


def test_cuda_execution_profile_environment_overrides_project_defaults() -> None:
    profile = resolve_cuda_execution_profile(
        environment={
            "VIBEQC_BENCHMARK_PARTITION": "accelerated",
            "VIBEQC_BENCHMARK_GRES": "gpu:a100:2",
            "VIBEQC_BENCHMARK_NODES": "2",
            "VIBEQC_BENCHMARK_NTASKS": "4",
            "VIBEQC_BENCHMARK_CPUS_PER_TASK": "8",
            "VIBEQC_BENCHMARK_TIME": "00:25:00",
            "VIBEQC_BENCHMARK_SRUN": "/opt/slurm/bin/srun",
        }
    )
    assert profile.partition == "accelerated"
    assert profile.gres == "gpu:a100:2"
    assert profile.nodes == 2
    assert profile.ntasks == 4
    assert profile.cpus_per_task == 8
    assert profile.slurm_time == "00:25:00"
    assert profile.srun == "/opt/slurm/bin/srun"


def test_explicit_execution_arguments_override_environment() -> None:
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


def test_empty_environment_scheduler_fields_disable_optional_flags() -> None:
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


def test_executor_uses_caller_time_as_default_but_environment_can_override() -> None:
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
        ({"VIBEQC_BENCHMARK_CPUS_PER_TASK": "many"}, "integer"),
    ],
)
def test_invalid_execution_environment_fails_closed(
    environment: typing.Any, match: typing.Any
) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_cuda_execution_profile(environment=environment)


@pytest.mark.parametrize(
    "explicit,environment_time,expected",
    [
        ("00:03:00", "00:20:00", "00:03:00"),
        (None, "00:20:00", "00:20:00"),
        (None, None, "00:10:00"),
    ],
)
def test_f_shell_cli_preserves_timeout_precedence(
    tmp_path: typing.Any,
    monkeypatch: typing.Any,
    explicit: typing.Any,
    environment_time: typing.Any,
    expected: typing.Any,
) -> None:
    """Trace the CLI argument into the numerical owner's actual executor."""
    import sys
    from types import SimpleNamespace

    from tools import validate_f_shells as cli

    class ReachedProfile(Exception):
        pass

    # No reference calculation is executed; optional PySCF is not a dependency
    # of scheduler argument parsing or this control-flow regression.
    monkeypatch.setitem(sys.modules, "pyscf", SimpleNamespace(__version__="unused"))
    if environment_time is None:
        monkeypatch.delenv("VIBEQC_BENCHMARK_TIME", raising=False)
    else:
        monkeypatch.setenv("VIBEQC_BENCHMARK_TIME", environment_time)
    original = CudaBenchmarkExecutor.from_environment

    def capture(cls: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> None:
        executor = original(*args, **kwargs)
        assert executor.slurm_time == expected
        raise ReachedProfile

    monkeypatch.setattr(CudaBenchmarkExecutor, "from_environment", classmethod(capture))
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/test-only/bin/nvcc")
    monkeypatch.setattr(
        cli, "catalog", lambda **kw: {"architecture": "sm_120", "rows": []}
    )
    monkeypatch.setattr(cli, "compile_matrix", lambda report, **kw: report)
    argv = ["validate_f_shells", "--tier=numerical", "--output", str(tmp_path / "out")]
    if explicit is not None:
        argv.append("--slurm-time=" + explicit)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ReachedProfile):
        cli.main()


@pytest.mark.parametrize("nodes,ntasks", [(1, 2), (2, 1), (2, 4)])
def test_distributed_requests_cannot_duplicate_single_process_benchmarks(
    nodes: typing.Any, ntasks: typing.Any
) -> None:
    profile = resolve_cuda_execution_profile(environment={}, nodes=nodes, ntasks=ntasks)
    assert profile.nodes == nodes and profile.ntasks == ntasks
    with pytest.raises(ValueError, match="one node and one task"):
        profile.wrap(["worker"])
    executor = CudaBenchmarkExecutor.from_environment(
        30, environment={}, nodes=nodes, ntasks=ntasks
    )
    with pytest.raises(ValueError, match="one node and one task"):
        executor.command(Path("worker"))
