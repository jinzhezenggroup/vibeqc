"""Shared native CPU executor for generated CC response TensorIR.

This is orchestration only: scientific equations remain owned by their TensorIR
program builders.  One executor caches compiled artifacts by exact program
identity and is intended to be shared across one bound CC/response lifecycle.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.tensor.cpu import NativeTensorProgram


class NativeCCTensorExecutor:
    """Compile/cache exact FP64 TensorIR programs through the #772 CPU backend."""

    backend = "native-cpu-tensorir"

    def __init__(
        self,
        *,
        max_bytes: int,
        max_work: int = 100_000_000_000,
        max_nodes: int = 8192,
        compiler: Path | None = None,
        cache: Path | None = None,
    ) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("native CC tensor max_bytes must be a positive integer")
        if type(max_work) is not int or max_work <= 0:
            raise ValueError("native CC tensor max_work must be a positive integer")
        if type(max_nodes) is not int or not 1 <= max_nodes <= 16384:
            raise ValueError("native CC tensor max_nodes must lie in [1, 16384]")
        compiler = (
            Path(os.environ.get("CXX", "c++")) if compiler is None else Path(compiler)
        )
        if cache is None:
            configured = os.environ.get("VIBEQC_TENSOR_CACHE")
            cache = (
                Path(configured) / "cc-response-cpu"
                if configured
                else Path.home() / ".cache" / "vibeqc" / "tensor-cpu" / "cc-response"
            )
        self.max_bytes = max_bytes
        self.max_work = max_work
        self.max_nodes = max_nodes
        self.compiler = CppCompilerAdapter(compiler)
        self.cache = Path(cache)
        self._programs: dict[str, NativeTensorProgram] = {}
        self._lock = threading.RLock()

    def execute(self, program: object, feeds: object) -> dict[str, object]:
        identity = getattr(program, "logical_hash", None)
        if not isinstance(identity, str) or not identity:
            raise TypeError(
                "native CC tensor execution requires an identified TensorIR Program"
            )
        with self._lock:
            compiled = self._programs.get(identity)
            if compiled is None:
                compiled = NativeTensorProgram(
                    program,
                    compiler=self.compiler,
                    cache=self.cache,
                    max_bytes=self.max_bytes,
                    max_work=self.max_work,
                    max_nodes=self.max_nodes,
                )
                self._programs[identity] = compiled
            return compiled.execute(feeds)

    @property
    def compiled_program_count(self) -> int:
        with self._lock:
            return len(self._programs)
