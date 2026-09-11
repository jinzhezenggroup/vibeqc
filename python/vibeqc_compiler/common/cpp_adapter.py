"""Explicit CPU C++ compilation using the same finite process adapter as CUDA."""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .backend import TargetInfo
from .compiler_process import CompileResult, run_compiler


@dataclass(frozen=True, slots=True)
class CppCompilerAdapter:
    """Compile native CPU shared libraries; selecting CUDA never uses this path."""

    cxx: Path
    compile_timeout: float = 300.0

    def __post_init__(self):
        executable = shutil.which(str(self.cxx))
        if executable is None:
            raise ValueError("the requested C++ compiler is unavailable")
        if (
            isinstance(self.compile_timeout, bool)
            or not math.isfinite(self.compile_timeout)
            or self.compile_timeout <= 0
        ):
            raise ValueError("compiler timeout must be finite and positive")
        # Preserve clang++/g++ invocation spelling: resolving a symlink to a
        # generic driver can change its language and standard-library defaults.
        object.__setattr__(self, "cxx", Path(executable).absolute())

    @property
    def target(self) -> TargetInfo:
        """Record the compiler's actual target triple without executing a binary."""
        triple = subprocess.check_output(
            [str(self.cxx), "-dumpmachine"], text=True, timeout=30
        ).strip()
        return TargetInfo("cpu", triple, None, 1, None)

    def compile_shared(
        self, source: Path, output: Path, *, includes=(), libraries=(), options=()
    ) -> CompileResult:
        """Compile explicit argv options and terminate the whole process tree on timeout."""
        return run_compiler(
            [
                str(self.cxx),
                "-std=c++17",
                "-O3",
                "-shared",
                "-fPIC",
                *(f"-I{path}" for path in includes),
                *options,
                str(source),
                *(f"-l{name}" for name in libraries),
                "-o",
                str(output),
            ],
            self.compile_timeout,
            label="C++",
        )
