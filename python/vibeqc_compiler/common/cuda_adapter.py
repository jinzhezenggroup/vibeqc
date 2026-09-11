"""CUDA compiler and benchmark execution adapters.

NVCC/PTXAS process handling and Slurm command construction live here so the
schedule search operates on CUDA target records rather than vendor CLI details.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .compiler_process import CompileResult as CudaCompileResult
from .compiler_process import run_compiler
from .cuda_target import CudaTargetInfo


@dataclass(frozen=True, slots=True)
class CudaCompilerAdapter:
    """Compile and link CUDA artifacts for one explicit target."""

    nvcc: Path
    target: CudaTargetInfo
    compile_timeout: float = 300.0

    def compile(self, source: Path, output: Path) -> CudaCompileResult:
        """Compile one translation unit and terminate all NVCC children on timeout."""

        command = [
            str(self.nvcc),
            "-std=c++17",
            f"-arch={self.target.architecture}",
            "-O3",
            "-Xptxas=-v",
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
    ) -> subprocess.CompletedProcess[str]:
        """Link compiled candidates and the target-probing driver."""

        return subprocess.run(
            [
                str(self.nvcc),
                "-std=c++17",
                f"-arch={self.target.architecture}",
                "-O3",
                str(driver),
                *(str(item) for item in objects),
                "-o",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


@dataclass(frozen=True, slots=True)
class CudaBenchmarkExecutor:
    """Run one CUDA benchmark locally or through a finite Slurm allocation."""

    timeout: int
    local: bool = False
    srun: str = "srun"
    partition: str = "main"
    gres: str = "gpu:1"
    slurm_time: str = "00:10:00"

    def command(self, executable: Path) -> list[str]:
        """Return the execution command without altering device visibility."""

        if self.local:
            return [str(executable)]
        return [
            self.srun,
            f"--partition={self.partition}",
            f"--gres={self.gres}",
            "--nodes=1",
            "--ntasks=1",
            f"--time={self.slurm_time}",
            str(executable),
        ]

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
