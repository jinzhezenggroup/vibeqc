"""CUDA compiler and benchmark execution adapters.

NVCC/PTXAS process handling and Slurm command construction live here so the
schedule search operates on CUDA target records rather than vendor CLI details.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .compiler_process import CompileResult as CudaCompileResult
from .compiler_process import run_compiler

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from .cuda_target import CudaTargetInfo


@dataclass(frozen=True, slots=True)
class CudaCompilerAdapter:
    """Compile and link CUDA artifacts for one explicit target."""

    nvcc: Path
    target: CudaTargetInfo
    compile_timeout: float = 300.0

    def compile(
        self,
        source: Path,
        output: Path,
        *,
        includes: tuple[Path, ...] = (),
        options: tuple[str, ...] = (),
        standard: str = "c++17",
    ) -> CudaCompileResult:
        """Compile one translation unit and terminate all NVCC children on timeout."""

        command = [
            str(self.nvcc),
            f"-std={standard}",
            f"-arch={self.target.architecture}",
            "-O3",
            "-Xptxas=-v",
            *(f"-I{path}" for path in includes),
            *options,
            "-c",
            str(source),
            "-o",
            str(output),
        ]
        return self._run_compiler(command)

    def compile_shared(
        self,
        source: Path,
        output: Path,
        *,
        includes: tuple[Path, ...] = (),
        libraries: tuple[str, ...] = (),
        options: tuple[str, ...] = (),
    ) -> CudaCompileResult:
        """Build a prepared native executor with the same finite process lifetime.

        Options are explicit argv entries, never shell text. Callers control
        scientific flags and libraries without changing legacy shell kernels.
        """
        return self._run_compiler(
            [
                str(self.nvcc),
                "-std=c++17",
                f"-arch={self.target.architecture}",
                "-O3",
                "-Xptxas=-v",
                "--shared",
                "-Xcompiler=-fPIC",
                *(f"-I{path}" for path in includes),
                *options,
                str(source),
                *(f"-l{name}" for name in libraries),
                "-o",
                str(output),
            ]
        )

    def link_shared_objects(
        self,
        objects: list[Path] | tuple[Path, ...],
        output: Path,
        *,
        libraries: tuple[str, ...] = (),
        options: tuple[str, ...] = (),
        standard: str = "c++17",
    ) -> CudaCompileResult:
        """Device-link relocatable CUDA objects into one shared runtime."""

        if not objects:
            raise ValueError("CUDA shared-object link requires at least one object")
        return self._run_compiler(
            [
                str(self.nvcc),
                f"-std={standard}",
                f"-arch={self.target.architecture}",
                "-O3",
                "-Xptxas=-v",
                "--relocatable-device-code=true",
                "--shared",
                "-Xcompiler=-fPIC",
                *options,
                *(str(item) for item in objects),
                *(f"-l{name}" for name in libraries),
                "-o",
                str(output),
            ]
        )

    def _run_compiler(self, command: list[str]) -> CudaCompileResult:
        """Bound NVCC and every child for either object or shared-library builds."""
        return run_compiler(command, self.compile_timeout, label="NVCC")

    def link(
        self,
        driver: Path,
        objects: list[Path],
        executable: Path,
        *,
        timeout: float = 300.0,
        includes: tuple[Path, ...] = (),
        options: tuple[str, ...] = (),
        standard: str = "c++17",
    ) -> subprocess.CompletedProcess[str]:
        """Link compiled candidates and the target-probing driver."""

        return subprocess.run(
            [
                str(self.nvcc),
                f"-std={standard}",
                f"-arch={self.target.architecture}",
                "-O3",
                *(f"-I{path}" for path in includes),
                str(driver),
                *(str(item) for item in objects),
                *options,
                "-o",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


@dataclass(frozen=True, slots=True)
class CudaExecutionProfile:
    """Portable scheduler resources for one CUDA validation/benchmark process."""

    local: bool = False
    srun: str = "srun"
    partition: str | None = "main"
    gres: str | None = "gpu:1"
    nodes: int = 1
    ntasks: int = 1
    slurm_time: str | None = "00:10:00"
    cpus_per_task: int | None = None

    def __post_init__(self) -> None:
        if self.nodes < 1 or self.ntasks < 1:
            raise ValueError("CUDA execution nodes/tasks must be positive")
        if self.cpus_per_task is not None and (
            type(self.cpus_per_task) is not int or self.cpus_per_task < 1
        ):
            raise ValueError(
                "CUDA execution cpus_per_task must be a positive integer or None"
            )
        if not self.srun.strip():
            raise ValueError("CUDA execution srun command must be non-empty")
        for name in ("partition", "gres", "slurm_time"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"CUDA execution {name} must be non-empty or None")

    def wrap(self, command: list[str]) -> list[str]:
        """Wrap an argv vector in the selected local or finite Slurm profile."""

        if not command:
            raise ValueError("CUDA execution command must be non-empty")
        if self.local:
            return list(command)
        # Every current consumer owns one result stream and one artifact path.
        # Multiple Slurm tasks would duplicate work and overwrite trial records,
        # not distribute one benchmark. Keep that request explicit and fail closed.
        if self.nodes != 1 or self.ntasks != 1:
            raise ValueError("CUDA benchmark launches require one node and one task")
        prefix = [self.srun]
        if self.partition is not None:
            prefix.append(f"--partition={self.partition}")
        if self.gres is not None:
            prefix.append(f"--gres={self.gres}")
        prefix.extend((f"--nodes={self.nodes}", f"--ntasks={self.ntasks}"))
        if self.cpus_per_task is not None:
            prefix.append(f"--cpus-per-task={self.cpus_per_task}")
        if self.slurm_time is not None:
            prefix.append(f"--time={self.slurm_time}")
        return [*prefix, *command]

    def to_dict(self) -> dict[str, object]:
        """Return stable provenance without scheduler-specific parsing."""

        return {
            "local": self.local,
            "srun": self.srun,
            "partition": self.partition,
            "gres": self.gres,
            "nodes": self.nodes,
            "ntasks": self.ntasks,
            "cpus_per_task": self.cpus_per_task,
            "slurm_time": self.slurm_time,
        }


def _environment_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def resolve_cuda_execution_profile(
    *,
    environment: Mapping[str, str] | None = None,
    local: bool | None = None,
    srun: str | None = None,
    partition: str | None = None,
    gres: str | None = None,
    nodes: int | None = None,
    ntasks: int | None = None,
    cpus_per_task: int | None = None,
    slurm_time: str | None = None,
    default_slurm_time: str | None = "00:10:00",
) -> CudaExecutionProfile:
    """Resolve explicit overrides over environment over project defaults.

    Empty optional scheduler strings in the environment disable that flag.
    The default requests one generic GPU without naming a model. Development
    clusters can select a concrete resource through explicit arguments or
    environment.
    """

    env = os.environ if environment is None else environment

    def text_value(key: str, explicit: str | None, default: str | None) -> str | None:
        if explicit is not None:
            return explicit
        if key in env:
            return env[key] or None
        return default

    def int_value(key: str, explicit: int | None, default: int) -> int:
        if explicit is not None:
            return explicit
        if key in env:
            try:
                return int(env[key])
            except ValueError as error:
                raise ValueError(f"{key} must be an integer") from error
        return default

    def optional_int_value(key: str, explicit: int | None) -> int | None:
        if explicit is not None:
            return explicit
        if key in env:
            try:
                return int(env[key])
            except ValueError as error:
                raise ValueError(f"{key} must be an integer") from error
        return None

    if local is None:
        resolved_local = (
            _environment_bool(env["VIBEQC_BENCHMARK_LOCAL"], "VIBEQC_BENCHMARK_LOCAL")
            if "VIBEQC_BENCHMARK_LOCAL" in env
            else False
        )
    else:
        resolved_local = local
    return CudaExecutionProfile(
        local=resolved_local,
        srun=text_value("VIBEQC_BENCHMARK_SRUN", srun, "srun") or "srun",
        partition=text_value("VIBEQC_BENCHMARK_PARTITION", partition, "main"),
        gres=text_value("VIBEQC_BENCHMARK_GRES", gres, "gpu:1"),
        nodes=int_value("VIBEQC_BENCHMARK_NODES", nodes, 1),
        ntasks=int_value("VIBEQC_BENCHMARK_NTASKS", ntasks, 1),
        cpus_per_task=optional_int_value(
            "VIBEQC_BENCHMARK_CPUS_PER_TASK", cpus_per_task
        ),
        slurm_time=text_value("VIBEQC_BENCHMARK_TIME", slurm_time, default_slurm_time),
    )


@dataclass(frozen=True, slots=True)
class CudaBenchmarkExecutor:
    """Run one CUDA benchmark locally or through a shared execution profile."""

    timeout: int
    local: bool = False
    srun: str = "srun"
    partition: str | None = "main"
    gres: str | None = "gpu:1"
    nodes: int = 1
    ntasks: int = 1
    slurm_time: str | None = "00:10:00"
    cpus_per_task: int | None = None

    @classmethod
    def from_environment(
        cls,
        timeout: int,
        *,
        local: bool | None = None,
        srun: str | None = None,
        partition: str | None = None,
        gres: str | None = None,
        nodes: int | None = None,
        ntasks: int | None = None,
        cpus_per_task: int | None = None,
        slurm_time: str | None = None,
        default_slurm_time: str | None = "00:10:00",
        environment: Mapping[str, str] | None = None,
    ) -> CudaBenchmarkExecutor:
        """Resolve one shared profile and adapt it to the benchmark executor."""

        profile = resolve_cuda_execution_profile(
            environment=environment,
            local=local,
            srun=srun,
            partition=partition,
            gres=gres,
            nodes=nodes,
            ntasks=ntasks,
            cpus_per_task=cpus_per_task,
            slurm_time=slurm_time,
            default_slurm_time=default_slurm_time,
        )
        return cls(
            timeout=timeout,
            local=profile.local,
            srun=profile.srun,
            partition=profile.partition,
            gres=profile.gres,
            nodes=profile.nodes,
            ntasks=profile.ntasks,
            cpus_per_task=profile.cpus_per_task,
            slurm_time=profile.slurm_time,
        )

    @property
    def profile(self) -> CudaExecutionProfile:
        """Expose the effective scheduler resources for provenance/tests."""

        return CudaExecutionProfile(
            local=self.local,
            srun=self.srun,
            partition=self.partition,
            gres=self.gres,
            nodes=self.nodes,
            ntasks=self.ntasks,
            cpus_per_task=self.cpus_per_task,
            slurm_time=self.slurm_time,
        )

    def command(self, executable: Path) -> list[str]:
        """Return the execution command without altering device visibility."""

        return self.profile.wrap([str(executable)])

    def run(
        self,
        executable: Path,
        environment: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        """Execute the benchmark with an explicit finite timeout."""

        return subprocess.run(
            self.command(executable),
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            env=environment,
        )
