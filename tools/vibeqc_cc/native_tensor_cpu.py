"""Shared native CPU executor for generated CC response TensorIR.

This is orchestration only: scientific equations remain owned by their TensorIR
program builders. One executor caches compiled artifacts by exact program
identity and is intended to be shared across one bound CC/response lifecycle.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.tensor.cpu import NativeTensorProgram
from vibeqc_compiler.tensor.cpu_bundle import NativeTensorProgramBundle


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
        self._bundled_programs: dict[str, tuple[NativeTensorProgramBundle, object]] = {}
        self._bundles: list[NativeTensorProgramBundle] = []
        self._lock = threading.RLock()

    def prewarm(self, programs: object) -> None:
        """Aggregate an exact not-yet-compiled program set into one native artifact."""
        try:
            candidates = tuple(programs)
        except TypeError as error:
            raise TypeError("native CC prewarm programs must be iterable") from error
        with self._lock:
            pending = []
            seen: set[str] = set()
            for program in candidates:
                identity = getattr(program, "logical_hash", None)
                if not isinstance(identity, str) or not identity:
                    raise TypeError(
                        "native CC prewarm requires identified TensorIR Programs"
                    )
                if identity in seen:
                    continue
                seen.add(identity)
                if identity in self._programs or identity in self._bundled_programs:
                    continue
                pending.append(program)
            if not pending:
                return
            bundle = NativeTensorProgramBundle(
                pending,
                compiler=self.compiler,
                cache=self.cache,
                max_bytes=self.max_bytes,
                max_work=self.max_work,
                max_nodes=self.max_nodes,
            )
            self._bundles.append(bundle)
            for program in pending:
                self._bundled_programs[program.logical_hash] = (bundle, program)

    def execute(self, program: object, feeds: object) -> dict[str, object]:
        identity = getattr(program, "logical_hash", None)
        if not isinstance(identity, str) or not identity:
            raise TypeError(
                "native CC tensor execution requires an identified TensorIR Program"
            )
        with self._lock:
            bundled = self._bundled_programs.get(identity)
            if bundled is not None:
                bundle, representative = bundled
                return bundle.execute(representative, feeds)
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
            return len(self._programs) + len(self._bundled_programs)

    @property
    def artifact_count(self) -> int:
        with self._lock:
            return len(self._programs) + len(self._bundles)

    @property
    def bundled_program_count(self) -> int:
        with self._lock:
            return len(self._bundled_programs)

    def _artifacts(self) -> tuple[object, ...]:
        return (
            *(compiled.artifact for compiled in self._programs.values()),
            *(bundle.artifact for bundle in self._bundles),
        )

    @property
    def binary_bytes(self) -> int:
        with self._lock:
            return sum(
                artifact.library.stat().st_size for artifact in self._artifacts()
            )

    @property
    def compile_seconds(self) -> float:
        """Compiler-reported cold build seconds for currently owned artifacts."""
        with self._lock:
            return float(
                sum(
                    float(artifact.metadata.get("compile_seconds", 0.0))
                    for artifact in self._artifacts()
                )
            )

    @property
    def artifact_libraries(self) -> tuple[Path, ...]:
        with self._lock:
            return tuple(artifact.library for artifact in self._artifacts())
