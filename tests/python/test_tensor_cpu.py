"""Executable CPU TensorIR subset: independent algebra and hostile ABI gates."""

import ctypes as ct
import typing
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.native_runtime import compile_runtime_bundle
from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    Symmetry,
    TensorSpec,
    add,
    broadcast,
    divide,
    einsum,
    execute,
    exp,
    gather,
    indexed_gather,
    input_tensor,
    multiply,
    reduce_sum,
    reshape,
    scaled_bilinear,
    scatter_add,
    segment_sum,
    slice_tensor,
    transpose,
)
from vibeqc_compiler.tensor.cpu import NativeTensorProgram, emit_cpu


def tensor(
    name: typing.Any,
    shape: typing.Any,
    dtype: typing.Any = "float64",
    symmetries: typing.Any = (),
) -> typing.Any:
    return input_tensor(
        name,
        TensorSpec(
            tuple(
                Index(chr(105 + k), IndexSpace("space" + str(n), "batch", n))
                for k, n in enumerate(shape)
            ),
            dtype=dtype,
            role="input",
            symmetries=tuple(symmetries),
        ),
    )


def native(
    program: typing.Any, tmp_path: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    return NativeTensorProgram(
        program, compiler=CppCompilerAdapter(Path("c++")), cache=tmp_path, **kwargs
    )


@pytest.mark.parametrize("size", [0, 3])
def test_reductions_einsums_and_detached_outputs(
    tmp_path: typing.Any, size: typing.Any
) -> None:
    a, b = tensor("a", (2, size)), tensor("b", (size, 4))
    contracted = einsum("ij,jk->ki", a, b, coefficient="-1/2")
    program = Program({"matrix": contracted, "total": reduce_sum(contracted, (0, 1))})
    executor = native(program, tmp_path)
    av = np.arange(2 * size, dtype=float).reshape(2, size)[:, ::-1]
    bv = np.arange(size * 4, dtype=float).reshape(size, 4)
    result = executor.execute({"a": av, "b": bv})
    np.testing.assert_array_equal(result["matrix"], -0.5 * (av @ bv).T)
    assert result["total"] == result["matrix"].sum()
    with pytest.raises(ValueError):
        result["matrix"].flags.writeable = True
    replay = native(Program.loads(program.dumps()), tmp_path)
    assert executor.identity == replay.identity
    assert executor.artifact.library == replay.artifact.library
    other = native(
        Program(dict(program.outputs), provenance={"different_plan": True}), tmp_path
    )
    assert executor.identity != other.identity


def test_two_tensor_programs_share_one_native_runtime_bundle(
    tmp_path: typing.Any,
) -> None:
    a = tensor("a", (3,))
    programs = (
        Program({"out": add(a, a)}),
        Program({"out": multiply(a, a)}),
    )
    paths = []
    resources = []
    for index, program in enumerate(programs):
        source, resource = emit_cpu(program, symbol=f"tensor_cpu_bundle_{index}")
        path = tmp_path / f"program_{index}.cpp"
        path.write_text(source)
        paths.append(path)
        resources.append(resource)

    header = asset_path("src/tensor/cpu_runtime.hpp")
    artifact = compile_runtime_bundle(
        CppCompilerAdapter(Path("c++")),
        tmp_path / "bundle-cache",
        paths,
        headers=(header,),
        options=("-ffp-contract=off", f"-I{header.parent}"),
    )
    library = ct.CDLL(str(artifact.library))
    values = np.arange(3, dtype=np.float64)
    ptr = lambda value: value.ctypes.data_as(ct.POINTER(ct.c_double))
    for index, expected in enumerate((2 * values, values * values)):
        call = getattr(library, f"tensor_cpu_bundle_{index}")
        call.argtypes = [
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.c_size_t,
        ]
        call.restype = ct.c_int
        output = np.empty(resources[index]["output_count"], dtype=np.float64)
        assert (
            call(ptr(values), values.size, ptr(output), output.size, 8 * 1024 * 1024)
            == 0
        )
        np.testing.assert_array_equal(output, expected)


def test_custom_native_entry_symbol_is_explicit_and_checked(
    tmp_path: typing.Any,
) -> None:
    a = tensor("a", (3,))
    program = Program({"a": add(a, a)})
    executor = NativeTensorProgram(
        program,
        compiler=CppCompilerAdapter(Path("c++")),
        cache=tmp_path,
        symbol="tensor_cpu_response_7",
    )
    np.testing.assert_array_equal(
        executor.execute({"a": np.arange(3, dtype=float)})["a"], [0.0, 2.0, 4.0]
    )
    source, _ = emit_cpu(program, symbol="tensor_cpu_response_8")
    assert 'extern "C" int tensor_cpu_response_8(' in source
    for symbol in ("", "7tensor", "tensor-cpu"):
        with pytest.raises(ValueError, match="C identifier"):
            emit_cpu(program, symbol=symbol)


def test_ordinary_pointwise_arithmetic(tmp_path: typing.Any) -> None:
    a, b = tensor("a", (4,)), tensor("b", (4,))
    executor = native(
        Program({"value": add(multiply(a, b), a, coefficients=(2, -3))}), tmp_path
    )
    av, bv = np.arange(4, dtype=float), np.arange(4, dtype=float) + 2
    np.testing.assert_array_equal(
        executor.execute({"a": av, "b": bv})["value"], 2 * av * bv - 3 * av
    )


def test_views_indexing_division_and_broadcast_match_interpreter(
    tmp_path: typing.Any,
) -> None:
    o, v = IndexSpace("o", "occupied", 3), IndexSpace("v", "virtual", 4)
    i, a = Index("i", o), Index("a", v)
    x = input_tensor("x", TensorSpec((i, a), role="input"))
    y = input_tensor("y", TensorSpec((i, a), role="input"))
    tile = slice_tensor(x, ((1, 3), (1, 4)))
    selected = gather(tile, 1, (2, 0, 2))
    reduced = reduce_sum(selected, (1,))
    batch = Index("batch", IndexSpace("batch", "batch", 2))
    expanded = broadcast(reduced, (batch, reduced.spec.indices[0]), (1,))
    permuted = transpose(selected, (1, 0))
    flat_axis = Index("flat", IndexSpace("flat", "batch", 6))
    flat = reshape(permuted, (flat_axis,))
    reordered = broadcast(x, (a, batch, i), (2, 0))
    program = Program(
        {
            "selected": selected,
            "expanded": expanded,
            "flat": flat,
            "reordered": reordered,
            "quotient": divide(x, y),
        }
    )
    xv = np.arange(24.0).reshape(3, 8)[:, ::-2]
    yv = np.arange(12.0).reshape(3, 4) + 1
    feeds = {"x": xv, "y": yv}
    expected = execute(program, feeds).outputs
    actual = native(program, tmp_path).execute(feeds)
    for output_name in program.outputs:
        np.testing.assert_array_equal(actual[output_name], expected[output_name])


def test_ragged_and_scaled_bilinear_match_interpreter(tmp_path: typing.Any) -> None:
    shell = Index("s", IndexSpace("shell", "shell", 3))
    orbital = Index("p", IndexSpace("orbital", "orbital", 5))
    segment = Index("g", IndexSpace("segment", "batch", 3))
    shells = input_tensor("shells", TensorSpec((shell,), role="input"))
    values = input_tensor("values", TensorSpec((orbital,), role="input"))
    mapping = (0, 0, 1, 2, 2)
    gathered = indexed_gather(shells, 0, mapping, orbital)
    scattered = scatter_add(values, 0, mapping, shell)
    segmented = segment_sum(values, 0, (0, 2, 2, 5), segment)
    nodes = [
        input_tensor(name, TensorSpec((orbital,), role="input"))
        for name in ("a", "b", "c", "d", "e", "f")
    ]
    safe = scaled_bilinear(*nodes)
    program = Program(
        {
            "gathered": gathered,
            "scattered": scattered,
            "segmented": segmented,
            "safe": safe,
        }
    )
    feeds = {
        "shells": np.array([2.0, -1.0, 4.0]),
        "values": np.array([1.0, 2.0, -3.0, 4.0, 5.0]),
        "a": np.array([1.0, 3.0, 5.0, 7.0, 11.0]),
        "b": np.array([2.0, -2.0, 4.0, 8.0, 3.0]),
        "c": np.array([0.5, 1.0, -2.0, 6.0, 4.0]),
        "d": np.array([1.0, -3.0, 2.0, 1.0, 5.0]),
        "e": np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
        "f": np.array([7.0, 8.0, 9.0, 10.0, 11.0]),
    }
    expected = execute(program, feeds).outputs
    executor = native(program, tmp_path)
    actual = executor.execute(feeds)
    for output_name in program.outputs:
        np.testing.assert_allclose(
            actual[output_name], expected[output_name], rtol=2e-15, atol=1e-15
        )
    bad = dict(feeds)
    bad["e"] = np.array([2.0, 0.0, 4.0, 5.0, 6.0])
    with pytest.raises(ValueError, match="native CPU tensor evaluation failed"):
        executor.execute(bad)


def test_dense_symmetry_is_validated_not_rejected(tmp_path: typing.Any) -> None:
    i = Index("i", IndexSpace("o", "occupied", 2))
    j = Index("j", i.space)
    x = input_tensor(
        "x",
        TensorSpec((i, j), role="input", symmetries=(Symmetry((1, 0)),)),
    )
    program = Program({"x": x, "t": transpose(x, (1, 0))})
    executor = native(program, tmp_path)
    symmetric = np.array([[1.0, 2.0], [2.0, 3.0]])
    result = executor.execute({"x": symmetric})
    np.testing.assert_array_equal(result["x"], symmetric)
    np.testing.assert_array_equal(result["t"], symmetric.T)
    with pytest.raises(ValueError, match="declared symmetry"):
        executor.execute({"x": np.array([[1.0, 2.0], [3.0, 4.0]])})


def test_scaled_bilinear_extreme_products_match_stable_interpreter(
    tmp_path: typing.Any,
) -> None:
    axis = Index("i", IndexSpace("case", "batch", 3))
    nodes = [
        input_tensor(name, TensorSpec((axis,), role="input"))
        for name in ("a", "b", "c", "d", "e", "f")
    ]
    program = Program({"out": scaled_bilinear(*nodes)})
    high = 1.0e300
    low = 1.0e-300
    feeds = {
        "a": np.array([high, low, high]),
        "b": np.array([high, low, low]),
        "c": np.array([high, low, 0.5]),
        "d": np.array([np.nextafter(high, 0.0), np.nextafter(low, 0.0), 1.0]),
        "e": np.array([high, low, 1.0e200]),
        "f": np.array([high, low, 1.0e-200]),
    }
    expected = execute(program, feeds).outputs["out"]
    actual = native(program, tmp_path).execute(feeds)["out"]
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0)


def test_preallocation_and_semantic_rejection(tmp_path: typing.Any) -> None:
    a = tensor("a", (3,))
    for program, message in (
        (Program({"a": tensor("a", (3,), "float32")}), "float64"),
        (
            Program({"a": exp(a)}),
            "unsupported",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            emit_cpu(program)
    with pytest.raises(ValueError, match="budget"):
        emit_cpu(Program({"a": a}), max_bytes=1)
    with pytest.raises(ValueError, match="budget"):
        emit_cpu(Program({"a": tensor("a", (10**12,))}))
    with pytest.raises(ValueError, match="budget"):
        emit_cpu(Program({"a": reduce_sum(a, (0,))}), max_work=1)
    with pytest.raises(TypeError, match="CPU compiler"):
        NativeTensorProgram(Program({"a": a}), compiler=object(), cache=tmp_path)
    for budget in (-1, True, 2**64):
        with pytest.raises(ValueError):
            emit_cpu(Program({"a": a}), max_bytes=budget)


def test_invalid_feeds_and_late_native_overflow_are_transactional(
    tmp_path: typing.Any,
) -> None:
    a = tensor("a", (3,))
    executor = native(Program({"a": multiply(a, a)}), tmp_path)
    for bad in (
        {},
        {"a": np.ones(2)},
        {"a": np.ones(3, dtype=np.float32)},
        {"a": np.array([1.0, np.nan, 2.0])},
    ):
        with pytest.raises(ValueError):
            executor.execute(bad)
    ptr = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_double))
    values = np.array([1.0, 2.0, 1e200])
    output = np.full(3, 42.0)
    for ni, no, budget in (
        (3, 3, executor.max_bytes),
        (2, 3, executor.max_bytes),
        (2**64 - 1, 3, executor.max_bytes),
        (3, 3, 0),
        (3, 2, executor.max_bytes),
    ):
        assert executor.call(ptr(values), ni, ptr(output), no, budget) != 0
        np.testing.assert_array_equal(output, 42.0)
    values[2] = np.nan
    assert executor.call(ptr(values), 3, ptr(output), 3, executor.max_bytes) != 0
    np.testing.assert_array_equal(output, 42.0)
    np.testing.assert_array_equal(
        executor.execute({"a": np.arange(3, dtype=float)})["a"], [0, 1, 4]
    )


def test_native_source_generation_does_not_probe_runtime_or_compilers() -> None:
    import subprocess
    import sys

    source = r"""
import importlib.abc
import sys
import ctypes
import subprocess
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'vibeqc', 'pyscf', 'cupy', 'torch'}:
            raise AssertionError('unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
def blocked(*args, **kwargs):
    raise AssertionError('generation probed a runtime or compiler')
ctypes.CDLL = blocked
subprocess.run = blocked
subprocess.Popen = blocked
subprocess.check_output = blocked
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.method.stationary_gradient import (
    StationaryGradientPlan, StationaryMeanField, SCF_POINT_MODEL,
)
from vibeqc_compiler.tensor.cpu import emit_cpu
from vibeqc_compiler.xc.grid_native import emit_grid_contraction
plan = StationaryGradientPlan(resolve_method('PBE'), StationaryMeanField(SCF_POINT_MODEL))
program = plan.integral_block('coulomb', terms=3).weights
first = emit_cpu(program)
assert first == emit_cpu(program)
assert 'tensor_cpu' in first[0]
first = emit_grid_contraction(3)
assert first == emit_grid_contraction(3)
assert 'grid_contract' in first
assert 'namespace vibeqc_grid_adjoint {' in first
assert 'grid_response_adjoint.hpp' not in first
"""
    subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
