"""Shared bounded CUDA executor for generated CC response TensorIR.

Scientific equations stay in their existing TensorIR builders. This owner only
plans, compiles, caches, executes, and releases one exact program at a time.
Prepared device allocations never overlap between calls, so a later response
stage can reuse the same explicit per-program budget without retaining an arena.
"""

from __future__ import annotations

import threading
import typing
from pathlib import Path

from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
from vibeqc_compiler.tensor.cuda_plan import plan_cuda


class CudaCCTensorExecutor:
    """Compile/cache exact FP64 CC TensorIR and execute it without CPU fallback."""

    backend = "cuda-fp64-ordinary-stream"

    def __init__(
        self,
        *,
        max_bytes: int,
        compiler: CudaCompilerAdapter,
        cache: Path,
        device: int = 0,
    ) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("CUDA CC tensor max_bytes must be a positive integer")
        if not isinstance(compiler, CudaCompilerAdapter):
            raise TypeError("CUDA CC tensor executor requires CudaCompilerAdapter")
        if not isinstance(cache, Path):
            raise TypeError("CUDA CC tensor cache must be a pathlib.Path")
        if type(device) is not int or device < 0:
            raise ValueError("CUDA CC tensor device must be a nonnegative ordinal")
        self.max_bytes = max_bytes
        self.compiler = compiler
        self.cache = cache
        self.device = device
        self._compiled: dict[str, tuple[object, object]] = {}
        self._metrics: dict[str, dict[str, typing.Any]] = {}
        self._lock = threading.RLock()
        self._plan_cuda = plan_cuda
        self._compile_cuda = compile_cuda
        self._PreparedCuda = PreparedCuda

    def _compile(self, program: object) -> tuple[object, object]:
        identity = getattr(program, "logical_hash", None)
        if not isinstance(identity, str) or not identity:
            raise TypeError(
                "CUDA CC tensor execution requires an identified TensorIR Program"
            )
        compiled = self._compiled.get(identity)
        if compiled is None:
            plan = self._plan_cuda(
                program,
                self.compiler.target,
                max_bytes=self.max_bytes,
            )
            artifact = self._compile_cuda(plan, self.compiler, self.cache)
            compiled = (plan, artifact)
            self._compiled[identity] = compiled
        return compiled

    def prewarm(self, programs: object) -> None:
        """Compile exact programs without preparing persistent device storage."""
        try:
            candidates = tuple(programs)
        except TypeError as error:
            raise TypeError("CUDA CC prewarm programs must be iterable") from error
        with self._lock:
            for program in candidates:
                self._compile(program)

    def execute(self, program: object, feeds: object) -> dict[str, object]:
        """Execute one program, closing its native CUDA arena before returning."""
        with self._lock:
            plan, artifact = self._compile(program)
            with self._PreparedCuda(plan, artifact, device=self.device) as prepared:
                result = prepared.execute(feeds)
            if result.backend != self.backend:
                raise RuntimeError(
                    "CUDA CC tensor backend changed; no CPU or alternate fallback allowed"
                )
            identity = program.logical_hash
            self._metrics[identity] = dict(result.metrics)
            return dict(result.outputs)

    @property
    def compiled_program_count(self) -> int:
        with self._lock:
            return len(self._compiled)

    @property
    def metrics(self) -> typing.Mapping[str, typing.Mapping[str, typing.Any]]:
        with self._lock:
            return {
                identity: dict(values) for identity, values in self._metrics.items()
            }
