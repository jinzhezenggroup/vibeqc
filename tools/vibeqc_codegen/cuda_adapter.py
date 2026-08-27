"""CUDA compiler and benchmark execution adapters.

NVCC/PTXAS process handling and Slurm command construction live here so the
schedule search operates on CUDA target records rather than vendor CLI details.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .cuda_target import CudaTargetInfo


@dataclass(frozen=True, slots=True)
class CudaCompileResult:
    """Compiler outcome including deterministic timeout diagnostics."""

    returncode: int
    timed_out: bool
    duration_seconds: float
    stdout: str
    stderr: str


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
        started = time.monotonic()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=self.compile_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
        duration = time.monotonic() - started
        if timed_out:
            stderr += (
                f"NVCC compilation timed out after {self.compile_timeout:g} seconds\n"
            )
        return CudaCompileResult(
            returncode=124 if timed_out else process.returncode,
            timed_out=timed_out,
            duration_seconds=duration,
            stdout=stdout,
            stderr=stderr,
        )

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
