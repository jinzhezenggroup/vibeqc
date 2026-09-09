"""Real-device TensorIR checks; opt in only inside an allocated CUDA job.

Example: srun -p main --gres=gpu:5090:1 --time=00:10:00 env
VIBEQC_TENSOR_CUDA_TEST=1 VIBEQC_NVCC=/path/to/nvcc python -m pytest ...
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from vibeqc.profiles import find_nvcc

from tools.vibeqc_codegen.cuda_adapter import CudaCompilerAdapter
from tools.vibeqc_codegen.cuda_target import cuda_target_info
from tools.vibeqc_tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    broadcast,
    constant,
    divide,
    einsum,
    gather,
    input_tensor,
    multiply,
    reduce_sum,
    reshape,
    slice_tensor,
    transpose,
    transpose_program,
)
from tools.vibeqc_tensor.cuda_execute import PreparedCuda, compile_cuda
from tools.vibeqc_tensor.cuda_plan import Reservations, TensorSchedule, plan_cuda
from tools.vibeqc_tensor.examples import example_cases
from tools.vibeqc_tensor.interpreter import execute

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)


@pytest.fixture(scope="module")
def compiler():
    nvcc = find_nvcc()
    if nvcc is None:
        pytest.fail("VIBEQC_TENSOR_CUDA_TEST requires a CUDA compiler")
    return CudaCompilerAdapter(
        nvcc, cuda_target_info(os.environ.get("VIBEQC_TENSOR_ARCH", "sm_120"))
    )


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return (
        Path(os.environ["VIBEQC_TENSOR_CACHE"])
        if "VIBEQC_TENSOR_CACHE" in os.environ
        else tmp_path_factory.mktemp("tensor-cuda")
    )


def check(program, feeds, compiler, cache, schedule=None, **options):
    schedule = TensorSchedule() if schedule is None else schedule
    expected = execute(program, feeds).outputs
    plan = plan_cuda(program, compiler.target, schedule=schedule, **options)
    artifact = compile_cuda(plan, compiler, cache)
    with PreparedCuda(plan, artifact) as prepared:
        for profile in (False, True, False):
            result = prepared.execute(feeds, profile=profile)
            for name in expected:
                np.testing.assert_allclose(
                    result.outputs[name], expected[name], atol=1e-11, rtol=1e-10
                )
            assert result.metrics["owned_device_bytes"] == plan.allocation_bytes
            assert result.metrics["provider_retained_bytes"] <= plan.provider_bytes
            assert result.metrics["predicted_peak_bytes"] <= plan.max_bytes
        return result


@pytest.mark.parametrize("case", ["diagonal", "named_inputs", "inactive_operand"])
def test_generated_vjp_review_regressions_on_cuda(case, compiler, cache):
    """Check generated adjoints against analytic results on the real backend."""
    if case == "named_inputs":
        spec = TensorSpec((), role="parameter", differentiable=True)
        first, second = input_tensor("x", spec), input_tensor("x", spec)
        primal = Program({"out": add(multiply(first, first), second)})
        generated = transpose_program(primal, ["out"], inputs=["x"])
        feeds = {"x": np.asarray(3.0), "bar_out": np.asarray(2.0)}
        output, expected = "bar_x", np.asarray(14.0)
    else:
        space = IndexSpace("o", "occupied", 3)
        matrix = input_tensor(
            "A",
            TensorSpec(
                (Index("i", space), Index("j", space)),
                role="parameter",
                differentiable=True,
            ),
        )
        if case == "diagonal":
            primal = Program({"out": einsum("ii->i", matrix)})
            generated = transpose_program(primal, ["out"], inputs=["A"])
            feeds = {"bar_out": np.array([2.0, 3.0, 4.0])}
            output, expected = "bar_A", np.diag(feeds["bar_out"])
        else:
            scalar = input_tensor(
                "x", TensorSpec((), role="parameter", differentiable=True)
            )
            primal = Program({"out": einsum("ii,->", matrix, scalar)})
            generated = transpose_program(primal, ["out"], inputs=["x"], max_elements=0)
            feeds = {"A": np.eye(3), "bar_out": np.asarray(2.0)}
            output, expected = "bar_x", np.asarray(6.0)
    result = check(generated.program, feeds, compiler, cache)
    np.testing.assert_array_equal(result.outputs[output], expected)


@pytest.mark.parametrize("case", example_cases(), ids=lambda c: c.name)
def test_examples_and_every_intermediate(case, compiler, cache):
    names = case.program.debug_names
    debug = Program({names[n]: n for n in case.program.live_nodes})
    check(debug, case.inputs, compiler, cache)
    for schedule in (
        TensorSchedule(),
        TensorSchedule(views=True, fuse=True),
        TensorSchedule(
            views=True,
            fuse=True,
            recompute=True,
            tile_m=1,
            tile_n=2,
            tile_k=1,
            direct_gemm=False,
        ),
    ):
        result = check(case.program, case.inputs, compiler, cache, schedule)
        np.testing.assert_allclose(
            result.outputs["value"], case.reference, atol=1e-11, rtol=1e-10
        )


@pytest.mark.parametrize(
    "expression,dimensions",
    [
        ("ik,kj->ij", {"i": 3, "j": 5, "k": 7}),
        ("ki,kj->ij", {"i": 3, "j": 5, "k": 7}),
        ("ik,jk->ij", {"i": 3, "j": 5, "k": 7}),
        ("ki,jk->ij", {"i": 3, "j": 5, "k": 7}),
        ("bik,bkj->bij", {"b": 3, "i": 3, "j": 5, "k": 7}),
        ("ibk,jkb->jbi", {"b": 3, "i": 3, "j": 5, "k": 7}),
        ("abef,ijef->ijab", {"a": 3, "b": 2, "e": 3, "f": 2, "i": 2, "j": 3}),
        ("ik,j->i", {"i": 3, "j": 5, "k": 7}),
        ("ii,ij->j", {"i": 3, "j": 5}),
        ("ij,jk,kl->il", {"i": 3, "j": 5, "k": 7, "l": 2}),
        ("i,j->ij", {"i": 3, "j": 5}),
        ("i,i->", {"i": 7}),
        (",i->i", {"i": 5}),
        ("ik,kj->ij", {"i": 3, "j": 5, "k": 0}),
        ("ik,kj->ij", {"i": 0, "j": 5, "k": 7}),
    ],
)
def test_einsum_layouts_and_partial_tiles(expression, dimensions, compiler, cache):
    from test_tensor_cuda_gemm import node_for

    node = node_for(expression, dimensions)
    rng = np.random.default_rng(146)
    feeds = {
        n.attrs["name"]: np.asarray(rng.normal(size=n.spec.shape), dtype=np.float64)
        for n in node.inputs
    }
    program = Program({"value": node})
    check(program, feeds, compiler, cache)
    check(
        program,
        feeds,
        compiler,
        cache,
        TensorSchedule(tile_m=2, tile_n=3, tile_k=2, direct_gemm=False),
        library_bytes=0,
    )


def test_views_gathers_general_reductions_and_empty_axes(compiler, cache):
    i, j = (
        Index("i", IndexSpace("i", "batch", 3)),
        Index("j", IndexSpace("j", "batch", 5)),
    )
    x = input_tensor("x", TensorSpec((i, j), role="input"))
    swapped = transpose(x, (1, 0))
    sliced = slice_tensor(swapped, ((1, 5), (0, 2)))
    selected = gather(sliced, 0, (3, 0, 3))
    flat = reshape(selected, (Index("q", IndexSpace("flat", "batch", 6)),))
    total = reduce_sum(flat, (0,))
    expanded = broadcast(total, (i, j), ())
    empty = slice_tensor(x, ((0, 0), (0, 5)))
    program = Program(
        {
            "result": add(multiply(x, expanded), x),
            "sum": total,
            "empty": reduce_sum(empty, (0,)),
            "scalar": constant("1/3"),
        }
    )
    feeds = {"x": np.arange(30, dtype=np.float64).reshape(3, 10)[:, ::-2]}
    for schedule in (
        TensorSchedule(),
        TensorSchedule(views=True, fuse=True),
        TensorSchedule(views=True, fuse=True, recompute=True),
    ):
        check(program, feeds, compiler, cache, schedule)


def test_errors_recovery_detachment_and_independent_contexts(compiler, cache):
    i = Index("i", IndexSpace("i", "batch", 5))
    x = input_tensor("x", TensorSpec((i,), role="input"))
    y = input_tensor("y", TensorSpec((i,), role="input"))
    program = Program({"a": divide(x, y), "b": divide(x, y)})
    plan = plan_cuda(
        program, compiler.target, schedule=TensorSchedule(views=True, fuse=True)
    )
    artifact = compile_cuda(plan, compiler, cache)
    with PreparedCuda(plan, artifact) as left, PreparedCuda(plan, artifact) as right:
        good = {"x": np.arange(5, dtype=np.float64), "y": np.ones(5, dtype=np.float64)}
        with pytest.raises(RuntimeError, match="division by zero"):
            left.execute({**good, "y": np.zeros(5, dtype=np.float64)})
        with pytest.raises(ValueError, match="non-finite"):
            left.execute({**good, "y": np.full(5, np.inf)})
        for scale in (1.0, 3.0, -2.0):
            with ThreadPoolExecutor(max_workers=2) as pool:
                a = pool.submit(left.execute, good)
                b = pool.submit(right.execute, {**good, "x": good["x"] * scale})
                first, second = a.result(), b.result()
            np.testing.assert_array_equal(first.outputs["a"], good["x"])
            np.testing.assert_array_equal(second.outputs["a"], good["x"] * scale)
            first.outputs["a"].fill(99)
            np.testing.assert_array_equal(first.outputs["b"], good["x"])
        with pytest.raises(ValueError, match="plan identity mismatch"):
            PreparedCuda(replace(plan, max_bytes=plan.max_bytes + 1), artifact)
    with pytest.raises(RuntimeError, match="closed"):
        left.execute(good)


def test_nonfinite_intermediate_and_minimum_budget(compiler, cache):
    i = Index("i", IndexSpace("i", "batch", 3))
    x = input_tensor("x", TensorSpec((i,), role="input"))
    square = multiply(x, x)
    program = Program({"z": add(square, square, coefficients=(1, -1))})
    for schedule in (TensorSchedule(), TensorSchedule(views=True, fuse=True)):
        plan = plan_cuda(
            program,
            compiler.target,
            schedule=schedule,
            reservations=Reservations(t=128, r=256, diis=512),
        )
        with pytest.raises(ValueError, match="infeasible"):
            plan_cuda(
                program,
                compiler.target,
                schedule=schedule,
                max_bytes=plan.peak_bytes - 1,
                reservations=plan.reservations,
            )
        with PreparedCuda(plan, compile_cuda(plan, compiler, cache)) as prepared:
            with pytest.raises(RuntimeError, match="non-finite"):
                prepared.execute({"x": np.full(3, 1e308)})
            np.testing.assert_array_equal(
                prepared.execute({"x": np.ones(3)}).outputs["z"], np.zeros(3)
            )


def test_shape_buckets_budget_and_concurrent_system_independence(compiler, cache):
    from tools.vibeqc_tensor.cuda_batch import PreparedTensorBatch

    plans, feeds = [], []
    for size in (3, 5, 3):
        i = Index("i", IndexSpace("occupied", "occupied", size))
        j = Index("j", IndexSpace("virtual", "virtual", size + 2))
        x = input_tensor("x", TensorSpec((i, j), role="parameter"))
        plans.append(plan_cuda(Program({"x2": multiply(x, x)}), compiler.target))
        feeds.append(
            {
                "x": np.arange(size * (size + 2), dtype=np.float64).reshape(
                    size, size + 2
                )
            }
        )
    budget = sum(p.peak_bytes for p in plans)
    with pytest.raises(ValueError, match="batch byte budget"):
        PreparedTensorBatch(plans, compiler, cache, max_bytes=budget - 1)
    with PreparedTensorBatch(plans, compiler, cache, max_bytes=budget) as batch:
        assert sorted(map(len, batch.buckets.values())) == [1, 2]
        for results in (batch.execute(feeds), batch.execute(feeds, workers=3)):
            for result, feed in zip(results, feeds, strict=True):
                np.testing.assert_array_equal(result.outputs["x2"], feed["x"] ** 2)
        changed = [{"x": feed["x"] + i + 1} for i, feed in enumerate(feeds)]
        for result, feed in zip(
            batch.execute(changed, workers=3), changed, strict=True
        ):
            np.testing.assert_array_equal(result.outputs["x2"], feed["x"] ** 2)


def test_architecture_mismatch_is_explicit_before_execution(compiler, cache):
    other = replace(
        compiler,
        target=cuda_target_info(
            "sm_80" if compiler.target.architecture != "sm_80" else "sm_86"
        ),
    )
    plan = plan_cuda(Program({"scalar": constant(3)}), other.target)
    with pytest.raises(ValueError, match="device architecture mismatch"):
        PreparedCuda(plan, compile_cuda(plan, other, cache))


def test_provider_allowance_guard_releases_a_rejected_handle(compiler, cache):
    case = example_cases()[0]
    plan = plan_cuda(case.program, compiler.target, library_bytes=0)
    artifact = compile_cuda(plan, compiler, cache)
    with PreparedCuda(plan, artifact) as prepared:
        measured = prepared.execute(case.inputs).metrics["provider_retained_bytes"]
    if measured == 0:
        pytest.skip("provider allocations fit existing allocator pages on this stack")
    # Deliberately bypass the CPU minimum solely to exercise native rollback
    # before tensor allocation; ordinary callers cannot obtain this plan.
    invalid = replace(plan, provider_bytes=0)
    with pytest.raises(RuntimeError, match="provider allowance"):
        PreparedCuda(invalid, compile_cuda(invalid, compiler, cache))
    with PreparedCuda(plan, artifact) as prepared:
        np.testing.assert_allclose(
            prepared.execute(case.inputs).outputs["value"],
            case.reference,
            atol=1e-11,
            rtol=1e-10,
        )
