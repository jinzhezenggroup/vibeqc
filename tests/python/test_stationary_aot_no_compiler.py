"""A packaged all-electron force request must not discover an NVCC compiler."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from vibeqc import _dft_gradient, _stationary_cuda
from vibeqc.batch import PreparedBatch
from vibeqc_compiler import dft
from vibeqc_compiler.common.cuda_target import cuda_target_info


def _evaluate_selector(node: ast.expr, values: dict[str, object]) -> object:
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        return any(bool(_evaluate_selector(value, values)) for value in node.values)
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Is)
        and len(node.comparators) == 1
    ):
        return _evaluate_selector(node.left, values) is _evaluate_selector(
            node.comparators[0], values
        )
    raise AssertionError(f"unsupported selector expression: {ast.dump(node)}")


def _artifact_selector(function_name: str, artifact_name: str) -> ast.IfExp:
    source = Path(__file__).resolve().parents[2] / "python/vibeqc/_stationary_cuda.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    assert len(functions) == 1
    function = functions[0]
    eager_sources = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "primitive_source"
            for target in node.targets
        )
    ]
    assert eager_sources == []
    assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.IfExp)
        and any(
            isinstance(target, ast.Name) and target.id == artifact_name
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    selector = assignments[0].value
    assert isinstance(selector, ast.IfExp)
    assert isinstance(selector.body, ast.Call)
    assert isinstance(selector.body.func, ast.Name)
    assert selector.body.func.id == "compile_stationary_cuda"
    assert isinstance(selector.orelse, ast.Call)
    assert isinstance(selector.orelse.func, ast.Name)
    assert selector.orelse.func.id == "load_stationary_aot_artifact"
    source_calls = {
        node.func.id
        for node in ast.walk(selector.body.args[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {"derivative_cuda_sources", "emit_first_derivative_cuda"} <= source_calls
    assert not {
        node.func.id
        for node in ast.walk(selector.orelse)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } & {"derivative_cuda_sources", "emit_first_derivative_cuda"}
    return selector


@pytest.mark.parametrize(
    ("function_name", "artifact_name"),
    [
        ("ensure", "stationary_artifact"),
        ("_complete_rks_cuda_gradient_diagnostic", "artifact"),
    ],
)
@pytest.mark.parametrize(
    ("aot_present", "ecp", "component_mode", "expected_jit"),
    [
        (True, False, False, False),
        (False, False, False, True),
        (True, True, False, True),
        (True, False, True, False),
        (False, True, True, True),
    ],
    ids=("sp-aot", "missing-aot", "ecp-jit", "component-aot", "combined-jit"),
)
def test_stationary_artifact_selector_keeps_source_emission_in_jit_branch(
    function_name: str,
    artifact_name: str,
    aot_present: bool,
    ecp: bool,
    component_mode: bool,
    expected_jit: bool,
) -> None:
    selector = _artifact_selector(function_name, artifact_name)
    values = {
        "aot_directory": object() if aot_present else None,
        "ecp": ecp,
        "component_mode": component_mode,
    }
    assert bool(_evaluate_selector(selector.test, values)) is expected_jit


@pytest.mark.parametrize("missing_artifact", (False, True))
def test_public_aot_force_does_not_probe_nvcc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing_artifact: bool
) -> None:
    class Basis:
        shells = ()

        def __enter__(self) -> object:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(dft, "NativeAO", lambda *args, **kwargs: Basis())
    source = SimpleNamespace(backend="cuda", hamiltonian="all-electron", close=Mock())
    state = SimpleNamespace(_source=source)
    monkeypatch.setattr(
        _dft_gradient.StationaryKsState, "from_native", lambda *args, **kwargs: state
    )
    target = cuda_target_info("sm_120")

    def calculate(*args: object, **kwargs: object) -> SimpleNamespace:
        assert kwargs["compiler"] is None
        assert kwargs["target"] is target
        assert kwargs["aot_directory"] == tmp_path
        if missing_artifact:
            raise FileNotFoundError("missing packaged stationary CUDA artifact")
        return SimpleNamespace(gradient=np.ones((2, 3)), work={"tensor_executions": 0})

    monkeypatch.setattr(
        _stationary_cuda, "complete_rks_cuda_gradient_diagnostic", calculate
    )
    batch = SimpleNamespace(
        _calculator=SimpleNamespace(
            _basis=object(),
            _representation_name="cartesian",
            _capabilities=SimpleNamespace(supported_properties={"energy", "forces"}),
        ),
        _stationary_cuda_execution=object(),
        _charges=[0],
        _multiplicities=[1],
        _library=SimpleNamespace(_name=str(tmp_path / "libvibeqc.so")),
        _stationary_cuda_compiler=lambda: pytest.fail("NVCC discovery before AOT load"),
        _stationary_cuda_target=lambda: target,
    )
    if missing_artifact:
        with pytest.raises(FileNotFoundError, match="missing packaged"):
            PreparedBatch._public_dft_cuda_force(batch, 0, ())
    else:
        force, work = PreparedBatch._public_dft_cuda_force(batch, 0, ())
        np.testing.assert_array_equal(force, -np.ones((2, 3)))
        assert work["tensor_executions"] == 0
    source.close.assert_called_once()


def test_public_d_shell_force_uses_component_aot_without_nvcc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Basis:
        shells = (SimpleNamespace(angular_momentum=2),)

        def __enter__(self) -> object:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(dft, "NativeAO", lambda *args, **kwargs: Basis())
    source = SimpleNamespace(backend="cuda", hamiltonian="all-electron", close=Mock())
    state = SimpleNamespace(_source=source)
    monkeypatch.setattr(
        _dft_gradient.StationaryKsState, "from_native", lambda *args, **kwargs: state
    )
    target = cuda_target_info("sm_120")

    def calculate(*args: object, **kwargs: object) -> SimpleNamespace:
        assert kwargs["compiler"] is None
        assert kwargs["target"] is target
        assert kwargs["aot_directory"] == tmp_path
        assert kwargs["native_grid_library"] == tmp_path / "libvibeqc.so"
        return SimpleNamespace(gradient=np.ones((2, 3)), work={"tensor_executions": 0})

    monkeypatch.setattr(
        _stationary_cuda, "complete_rks_cuda_gradient_diagnostic", calculate
    )
    batch = SimpleNamespace(
        _calculator=SimpleNamespace(
            _basis=object(),
            _representation_name="spherical",
            _capabilities=SimpleNamespace(supported_properties={"energy", "forces"}),
        ),
        _stationary_cuda_execution=object(),
        _charges=[0],
        _multiplicities=[1],
        _library=SimpleNamespace(_name=str(tmp_path / "libvibeqc.so")),
        _stationary_cuda_compiler=lambda: pytest.fail(
            "NVCC discovery before component AOT load"
        ),
        _stationary_cuda_target=lambda: target,
    )

    force, work = PreparedBatch._public_dft_cuda_force(batch, 0, ())

    np.testing.assert_array_equal(force, -np.ones((2, 3)))
    assert work["tensor_executions"] == 0
    source.close.assert_called_once()
