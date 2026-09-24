"""Cached CC CUDA plans must not outlive their configured resource identity."""

from pathlib import Path
from types import SimpleNamespace
from typing import Self

import pytest
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info

from tools.vibeqc_cc.native_tensor_cuda import CudaCCTensorExecutor


def _executor(tmp_path: Path) -> tuple[CudaCCTensorExecutor, list[str]]:
    compiler = CudaCompilerAdapter(Path("nvcc"), cuda_target_info("sm_120"))
    executor = CudaCCTensorExecutor(max_bytes=8192, compiler=compiler, cache=tmp_path)
    events: list[str] = []

    def plan(program: object, target: object, *, max_bytes: int) -> object:
        del program
        events.append("plan")
        return SimpleNamespace(peak_bytes=max_bytes, target=target)

    def compile_plan(*args: object) -> object:
        del args
        events.append("compile")
        return object()

    class Prepared:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            events.append("prepare")

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            del args
            events.append("close")

        def execute(self, feeds: object) -> object:
            del feeds
            events.append("execute")
            return SimpleNamespace(
                backend="cuda-fp64-ordinary-stream", metrics={}, outputs={"value": 7}
            )

    executor._plan_cuda = plan
    executor._compile_cuda = compile_plan
    executor._PreparedCuda = Prepared
    return executor, events


@pytest.mark.parametrize(
    "field", ["max_bytes", "compiler", "cache", "device", "backend"]
)
@pytest.mark.parametrize("method", ["execute", "prewarm"])
def test_changed_configuration_rejected_before_cached_plan_use(
    tmp_path: Path, field: str, method: str
) -> None:
    executor, events = _executor(tmp_path)
    program = SimpleNamespace(logical_hash="fixed-program")
    executor.prewarm([program])
    original = getattr(executor, field)
    changes = {
        "max_bytes": 1,
        "compiler": CudaCompilerAdapter(
            Path("another-nvcc"), cuda_target_info("sm_80")
        ),
        "cache": tmp_path / "other-cache",
        "device": 1,
        "backend": "numpy-cpu-interpreter",
    }
    setattr(executor, field, changes[field])
    with pytest.raises(RuntimeError, match="configuration changed"):
        if method == "execute":
            executor.execute(program, {})
        else:
            executor.prewarm([program])
    assert events == ["plan", "compile"]
    assert executor.compiled_program_count == 1
    assert executor.metrics == {}

    # Restore the exact original admission: its compiled plan may be reused,
    # but every execution must still own and close a distinct native arena.
    setattr(executor, field, original)
    assert executor.execute(program, {}) == {"value": 7}
    assert events == ["plan", "compile", "prepare", "execute", "close"]


def test_unchanged_configuration_reuses_plan_not_arena(tmp_path: Path) -> None:
    executor, events = _executor(tmp_path)
    program = SimpleNamespace(logical_hash="fixed-program")
    assert executor.execute(program, {}) == executor.execute(program, {})
    assert events == [
        "plan",
        "compile",
        "prepare",
        "execute",
        "close",
        "prepare",
        "execute",
        "close",
    ]
