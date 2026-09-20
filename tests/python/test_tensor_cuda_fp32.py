"""Typed CUDA plans plus opt-in real FP32 execution; no molecular promotion."""

import os
import typing
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    Symmetry,
    TensorSpec,
    add,
    broadcast,
    constant,
    divide,
    einsum,
    execute,
    gather,
    input_tensor,
    multiply,
    reduce_sum,
    reshape,
    slice_tensor,
    transpose,
)
from vibeqc_compiler.tensor.cuda_dtype import compile_options, scalar_type
from vibeqc_compiler.tensor.cuda_emit import emit_cuda
from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
from vibeqc_compiler.tensor.cuda_gemm import gemm_contract
from vibeqc_compiler.tensor.cuda_plan import VALIDATION_BYTES, TensorSchedule, plan_cuda
from vibeqc_compiler.tensor.cuda_resident import PreparedResident, compile_resident
from vibeqc_compiler.tensor.cuda_resident_emit import resident_source
from vibeqc_compiler.tensor.resources import tensor_resource_choices

TARGET = cuda_target_info("sm_120")
DEVICE = pytest.mark.skipif(
    os.environ.get("VIBEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires an explicitly allocated CUDA device",
)


def tensor(
    name: typing.Any,
    shape: typing.Any = (),
    dtype: typing.Any = "float32",
    *,
    symmetries: typing.Any = (),
) -> typing.Any:
    return input_tensor(
        name,
        TensorSpec(
            tuple(
                Index(f"i{i}", IndexSpace(f"d{n}", "batch", n))
                for i, n in enumerate(shape)
            ),
            dtype=dtype,
            role="parameter",
            differentiable=True,
            symmetries=symmetries,
        ),
    )


def contraction(
    expression: typing.Any, dtype: typing.Any = "float32", dimensions: typing.Any = None
) -> typing.Any:
    dims = {"b": 2, "i": 3, "j": 5, "k": 7} if dimensions is None else dimensions
    inputs = [
        input_tensor(
            f"x{i}",
            TensorSpec(
                tuple(
                    Index(f"a{j}", IndexSpace(label, "batch", dims[label]))
                    for j, label in enumerate(labels)
                ),
                dtype=dtype,
                role="input",
            ),
        )
        for i, labels in enumerate(expression.split("->")[0].split(","))
    ]
    return Program({"out": einsum(expression, *inputs, coefficient=Fraction(-3, 4))})


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_plan_storage_precision_constants_and_resident_spans(
    dtype: typing.Any,
) -> None:
    x = tensor("x", (513,), dtype)
    program = Program({"out": add(x, x)})
    plan = plan_cuda(program, TARGET)
    width = np.dtype(dtype).itemsize
    assert plan.host_bytes == 2 * 513 * width + VALIDATION_BYTES
    assert plan.estimated_traffic_bytes == 3 * 513 * width
    assert plan.precision == f"fp{8 * width}"
    assert plan.to_payload()["steps"][0]["itemsize"] == width
    code = emit_cuda(plan)
    assert f"__{scalar_type(dtype).prefix}add_rn" in code
    assert f"{513 * width}ULL, cudaMemcpyDeviceToHost" in code
    source = resident_source(plan)
    for step in plan.steps:
        assert f"{{{step.offset}ULL,{513 * width}ULL}}" in source
    choices = tensor_resource_choices(program, TARGET)
    assert choices.request.identity.precision == plan.precision
    assert plan.identity == plan_cuda(Program.loads(program.dumps()), TARGET).identity
    if dtype == "float32":
        assert "--ftz=false" in compile_options(plan)
        assert "__CUDA_FTZ" in source
    for coefficient in ((1, 3), (10**400, 10**400)):
        scalar = scalar_type(dtype)
        assert scalar.coefficient(coefficient) == float(
            np.dtype(dtype).type(float(Fraction(*coefficient)))
        )
    with pytest.raises(ValueError, match="finite FP"):
        plan_cuda(
            Program(
                {"huge": constant(10**400, TensorSpec(dtype=dtype, role="constant"))}
            ),
            TARGET,
        )


def test_dtype_identity_mixed_components_and_integer_tables() -> None:
    x32, x64 = tensor("s", (513,)), tensor("d", (513,), "float64")
    small, large = [plan_cuda(Program({"x": x}), TARGET) for x in (x32, x64)]
    assert small.identity != large.identity
    assert small.arena_bytes < large.arena_bytes
    with pytest.raises(ValueError, match="dtype"):
        add(x32, x64)
    program = Program({"s": gather(x32, 0, (1, 0, 1)), "d": add(x64, x64)})
    plan = plan_cuda(program, TARGET)
    assert plan.precision == "typed-fp32-fp64"
    source = emit_cuda(plan)
    assert "reinterpret_cast<const float*>" in source
    assert "reinterpret_cast<const double*>" in source
    assert "24ULL, cudaMemcpyHostToDevice" in source  # three int64 indices
    assert "12ULL, cudaMemcpyDeviceToHost" in source
    assert "4104ULL, cudaMemcpyDeviceToHost" in source


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gemm_panels_and_precision_admission(dtype: typing.Any) -> None:
    program = contraction("ik,kj->ij", dtype)
    g = gemm_contract(program.outputs["out"])
    assert g.panel_bytes(2, 3, 4) == np.dtype(dtype).itemsize * (2 * 4 + 4 * 3 + 2 * 3)
    plan = plan_cuda(program, TARGET, schedule=TensorSchedule(direct_gemm=False))
    assert plan.steps[-1].gemm == "packed"
    assert ("CUBLAS_PEDANTIC_MATH" in emit_cuda(plan)) == (dtype == "float32")
    with pytest.raises(ValueError, match="budget"):
        plan_cuda(program, TARGET, max_bytes=1)


@pytest.fixture(scope="module")
def gpu(tmp_path_factory: typing.Any) -> typing.Any:
    from vibeqc.profiles import find_nvcc

    nvcc = find_nvcc()
    assert nvcc is not None, "explicit CUDA tests require an NVCC compiler"
    return CudaCompilerAdapter(nvcc, TARGET), Path(
        os.environ.get("VIBEQC_TENSOR_CACHE", str(tmp_path_factory.mktemp("fp32-cuda")))
    )


def prepare(
    program: typing.Any,
    gpu: typing.Any,
    schedule: typing.Any = None,
    resident: typing.Any = False,
) -> typing.Any:
    compiler, cache = gpu
    schedule = TensorSchedule() if schedule is None else schedule
    plan = plan_cuda(program, compiler.target, schedule=schedule)
    compile_fn, cls = (
        (compile_resident, PreparedResident)
        if resident
        else (compile_cuda, PreparedCuda)
    )
    return cls(plan, compile_fn(plan, compiler, cache))


@DEVICE
@pytest.mark.parametrize(
    "expression",
    [
        "ik,kj->ij",
        "ki,kj->ij",
        "ik,jk->ij",
        "ki,jk->ij",
        "bik,bkj->bij",
        "ibk,jkb->jbi",
    ],
)
@pytest.mark.parametrize("direct", [False, True])
def test_fp32_sgemm_transposes_batches_and_partial_panels(
    gpu: typing.Any, expression: typing.Any, direct: typing.Any
) -> None:
    program = contraction(expression)
    rng = np.random.default_rng(482)
    feeds = {
        n.attrs["name"]: rng.normal(size=n.spec.shape).astype(np.float32)
        for n in program.live_nodes
        if n.op == "input"
    }
    # Binary input values, independently contracted in FP64, then rounded once.
    expected = (
        -0.75
        * np.einsum(expression, *[feeds[f"x{i}"].astype(np.float64) for i in range(2)])
    ).astype(np.float32)
    schedule = TensorSchedule(direct_gemm=direct, tile_m=2, tile_n=3, tile_k=4)
    with prepare(program, gpu, schedule) as prepared:
        for profile in (False, True):
            actual = prepared.execute(feeds, profile=profile)
            assert actual.outputs["out"].dtype == np.float32
            assert actual.metrics["precision"] == "fp32"
            assert actual.backend == "cuda-fp32-ordinary-stream"
            np.testing.assert_allclose(
                actual.outputs["out"], expected, rtol=3e-6, atol=2e-6
            )
        with pytest.raises(ValueError, match="float32"):
            prepared.execute({k: a.astype(np.float64) for k, a in feeds.items()})


@DEVICE
@pytest.mark.parametrize("optimized", [False, True])
def test_fp32_primitives_views_general_contractions_and_empty_outputs(
    gpu: typing.Any, optimized: typing.Any
) -> None:
    x, y = tensor("x", (3, 5)), tensor("y", (3, 5))
    rows = reduce_sum(multiply(x, y), (1,))
    outputs = {
        "quotient": divide(add(x, y, coefficients=(Fraction(1, 3), -2)), y),
        "transposed": transpose(x, (1, 0)),
        "slice": slice_tensor(x, ((1, 3), (0, 4))),
        "gather": gather(x, 1, (4, 0, 4)),
        "empty": slice_tensor(x, ((0, 0), (0, 5))),
        "expanded": broadcast(rows, x.spec.indices, (0,)),
        "general": einsum("ij,ij,ij->i", x, y, x),
        "reshape": reshape(x, (Index("flat", IndexSpace("flat", "batch", 15)),)),
    }
    feeds = {
        "x": (np.arange(30, dtype=np.float32).reshape(3, 10) / 16)[:, ::2],
        "y": np.full((3, 5), 2, np.float32),
    }
    program = Program(outputs)
    schedule = TensorSchedule(views=optimized, fuse=optimized, recompute=optimized)
    expected = execute(program, feeds).outputs
    with prepare(program, gpu, schedule) as prepared:
        result = prepared.execute(feeds)
        for name in expected:
            np.testing.assert_allclose(
                result.outputs[name], expected[name], rtol=2e-6, atol=2e-6
            )
            assert result.outputs[name].dtype == np.float32
        with pytest.raises(RuntimeError, match="division by zero"):
            prepared.execute({**feeds, "y": np.zeros((3, 5), np.float32)})
        with pytest.raises(ValueError, match="non-finite"):
            prepared.execute({**feeds, "y": np.full((3, 5), np.nan, np.float32)})
        prepared.execute(feeds)  # recovery on the same handle
    empty_k = contraction("ik,kj->ij", dimensions={"i": 3, "j": 5, "k": 0})
    with prepare(empty_k, gpu) as prepared:
        actual = prepared.execute(
            {"x0": np.empty((3, 0), np.float32), "x1": np.empty((0, 5), np.float32)}
        )
        np.testing.assert_array_equal(
            actual.outputs["out"], np.zeros((3, 5), np.float32)
        )


@DEVICE
def test_real_fp32_rounding_denormals_and_mixed_component_boundaries(
    gpu: typing.Any,
) -> None:
    x, y, z = (tensor(n, (3,)) for n in "xyz")
    d = tensor("double_input", (3,), "float64")
    program = Program({"single": add(x, y, z), "double": add(d, d)})
    feeds = {
        "x": np.array([2**24, np.finfo(np.float32).smallest_subnormal, 1], np.float32),
        "y": np.array([1, 0, 2**-23], np.float32),
        "z": np.array([-(2**24), 0, 0], np.float32),
        "double_input": np.array([1, 2, 3], np.float64),
    }
    with prepare(program, gpu) as prepared:
        actual = prepared.execute(feeds).outputs
        np.testing.assert_array_equal(
            actual["single"],
            [0, np.finfo(np.float32).smallest_subnormal, np.float32(1 + 2**-23)],
        )
        assert (
            actual["single"].dtype == np.float32
            and actual["double"].dtype == np.float64
        )
    # A product distinguishable from TF32 input truncation and from FP16.
    program = contraction("ik,kj->ij", dimensions={"i": 1, "j": 1, "k": 1})
    a = np.array([[1 + 2**-12]], np.float32)
    with prepare(program, gpu) as prepared:
        actual = prepared.execute({"x0": a, "x1": a}).outputs["out"]
        expected = np.float32(-0.75) * np.float32(a * a)
        np.testing.assert_array_equal(actual, expected)


@DEVICE
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_resident_typed_spans_transfers_and_validation(
    gpu: typing.Any, dtype: typing.Any
) -> None:
    x = tensor("x", (3, 3), dtype, symmetries=(Symmetry((1, 0)),))
    program = Program({"out": multiply(x, x)})
    data = np.eye(3, dtype=dtype)
    with prepare(program, gpu, resident=True) as prepared:
        with pytest.raises(ValueError, match=dtype):
            prepared.upload(
                {"x": data.astype("float64" if dtype == "float32" else "float32")}
            )
        bad = data.copy()
        bad[0, 1] = 0.25
        with pytest.raises(ValueError, match="symmetry"):
            prepared.upload({"x": bad})
        prepared.upload({"x": data})
        outputs, metrics = prepared.run()
        assert outputs["out"].dtype == np.dtype(dtype)
        np.testing.assert_array_equal(outputs["out"].to_host(), data)
        assert prepared.transfers["h2d_bytes"] == data.nbytes
        assert prepared.transfers["d2h_bytes"] == data.nbytes + 4
        assert metrics["precision"] == ("fp32" if dtype == "float32" else "fp64")
        prepared.run()
        with pytest.raises(RuntimeError, match="stale"):
            outputs["out"].to_host()


@pytest.mark.parametrize(
    "flags",
    [
        "--use_fast_math",
        "--ftz=true",
        "--ftz true",
        "--prec-div=false",
        "--prec-sqrt false",
        "--fmad=true",
        "--fmad true",
        "-fmad=1",
    ],
)
def test_strict_fp32_rejects_arithmetic_environment_overrides(
    monkeypatch: typing.Any, flags: typing.Any
) -> None:
    plan = plan_cuda(Program({"x": tensor("x")}), TARGET)
    monkeypatch.setenv("NVCC_APPEND_FLAGS", flags)
    with pytest.raises(ValueError, match="FP32 TensorIR"):
        compile_options(plan)


def test_fp32_tuning_requires_independent_promotion_gates() -> None:
    from vibeqc_compiler.tensor.cuda_tune import tune_cuda

    plan = plan_cuda(Program({"x": tensor("x")}), TARGET)
    with pytest.raises(ValueError, match="qualified only for FP64"):
        tune_cuda(plan, None, [], Path("unused"))
