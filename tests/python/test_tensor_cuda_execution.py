"""Real-device TensorIR checks; opt in only inside an allocated CUDA job.

Example: srun -p main --gres=gpu:5090:1 --time=00:10:00 env
VIBEQC_TENSOR_CUDA_TEST=1 VIBEQC_NVCC=/path/to/nvcc python -m pytest ...
"""

from __future__ import annotations

import os
import typing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from vibeqc.profiles import find_nvcc
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import (
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
from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
from vibeqc_compiler.tensor.cuda_plan import Reservations, TensorSchedule, plan_cuda
from vibeqc_compiler.tensor.examples import example_cases
from vibeqc_compiler.tensor.interpreter import execute

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)


@pytest.fixture(scope="module")
def compiler() -> typing.Any:
    nvcc = find_nvcc()
    if nvcc is None:
        pytest.fail("VIBEQC_TENSOR_CUDA_TEST requires a CUDA compiler")
    return CudaCompilerAdapter(
        nvcc, cuda_target_info(os.environ.get("VIBEQC_TENSOR_ARCH", "sm_120"))
    )


@pytest.fixture(scope="module")
def cache(tmp_path_factory: typing.Any) -> typing.Any:
    return (
        Path(os.environ["VIBEQC_TENSOR_CACHE"])
        if "VIBEQC_TENSOR_CACHE" in os.environ
        else tmp_path_factory.mktemp("tensor-cuda")
    )


def test_two_tensor_providers_share_one_global_budget(
    compiler: typing.Any, cache: typing.Any
) -> None:
    """A retained neighbor forces an executable recomputation alternative."""
    from vibeqc.resources import ResourceBudget, ResourceSession, plan_resources
    from vibeqc_compiler.tensor.resources import tensor_resource_choices

    index = Index("i", IndexSpace("axis", "batch", 8192))
    x = input_tensor("x", TensorSpec((index,), role="input"))
    program = Program(
        {
            f"r{i}": reduce_sum(add(x, x, coefficients=(1, i + 1)), (0,))
            for i in range(6)
        }
    )
    primary = tensor_resource_choices(program, compiler.target, name="primary")
    neighbor = tensor_resource_choices(
        Program({"x": x}), compiler.target, name="neighbor"
    )
    smallest = min((p for _, p in primary.plans), key=lambda p: p.device_bytes)
    neighbor_plan = neighbor.plans[0][1]
    assert smallest.schedule.recompute
    budget = ResourceBudget(
        device_bytes=smallest.device_bytes + neighbor_plan.device_bytes,
        host_bytes=smallest.host_bytes + neighbor_plan.host_bytes,
    )
    global_plan = plan_resources(
        [primary.request, neighbor.request], budget
    ).require_feasible()
    selected = primary.selected(global_plan)
    assert selected.schedule.recompute
    feeds = {"x": np.linspace(-0.25, 0.75, 8192)}
    with ResourceSession(
        global_plan,
        {
            "neighbor": neighbor.factory(compiler, cache),
            "primary": primary.factory(compiler, cache),
        },
    ) as session:
        session.advance(0)
        other, prepared = session.provider("neighbor"), session.provider("primary")
        other_result = other.execute(feeds)
        result = prepared.execute(feeds)
        for i in range(6):
            np.testing.assert_allclose(
                result.outputs[f"r{i}"], (i + 2) * feeds["x"].sum(), rtol=1e-11
            )
        np.testing.assert_array_equal(other_result.outputs["x"], feeds["x"])
        for space in ("device", "host"):
            measured = sum(
                r.metrics[f"tracked_{space}_bytes"] for r in (other_result, result)
            )
            assert measured <= global_plan.peak_bytes[space]
        assert result.metrics["resource_plan_id"] == global_plan.identity


def test_actual_cuda_allocation_failure_is_typed_and_exhausted_plan_is_recorded(
    compiler: typing.Any, cache: typing.Any
) -> None:
    """An impossible native allocation tests rollback without filling GPU RAM."""
    from vibeqc import (
        ResourceAllocationError,
        ResourceBudget,
        ResourceSession,
        plan_resources,
    )
    from vibeqc_compiler.tensor.resources import tensor_resource_choices

    program = Program({"scalar": constant(3)})
    choices = tensor_resource_choices(
        program,
        compiler.target,
        reservations=Reservations(concurrent=1 << 48),
        sub_budget_bytes=1 << 49,
    )
    plan = plan_resources([choices.request], ResourceBudget())
    with ResourceSession(plan, {"tensor": choices.factory(compiler, cache)}) as session:
        with pytest.raises(ResourceAllocationError) as failure:
            session.advance(0)
        assert failure.value.space == "device:0"
        assert len(session.fallbacks) == 1
        assert session.fallbacks[0]["to_plan"] is None
    ordinary = tensor_resource_choices(program, compiler.target)
    ordinary_plan = plan_resources([ordinary.request], ResourceBudget())
    with ResourceSession(
        ordinary_plan, {"tensor": ordinary.factory(compiler, cache)}
    ) as session:
        session.advance(0)
        assert session.provider("tensor").execute({}).outputs["scalar"] == 3


@pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="also requires a CUDA-linked HF library",
)
def test_direct_hf_and_tensor_share_one_executable_resource_plan(
    compiler: typing.Any, cache: typing.Any
) -> None:
    from vibeqc import Calculator, ResourceBudget, ResourceSession, plan_resources
    from vibeqc_compiler.tensor.resources import tensor_resource_choices

    atoms = [(1, (0, 0, -0.7)), (1, (0, 0, 0.7))]
    calculator = Calculator(device="cuda")
    hf = calculator._resource_request([atoms])
    index = Index("i", IndexSpace("axis", "batch", 8192))
    x = input_tensor("x", TensorSpec((index,), role="input"))
    program = Program(
        {
            f"r{i}": reduce_sum(add(x, x, coefficients=(1, i + 1)), (0,))
            for i in range(6)
        }
    )
    tensor = tensor_resource_choices(program, compiler.target)
    smallest = min((p for _, p in tensor.plans), key=lambda p: p.device_bytes)
    hf_plan = plan_resources([hf], ResourceBudget()).require_feasible()
    budget = ResourceBudget(
        device_bytes=hf_plan.peak_bytes["device"] + smallest.device_bytes,
        host_bytes=hf_plan.peak_bytes["host"] + smallest.host_bytes,
    )
    plan = plan_resources([hf, tensor.request], budget).require_feasible()
    assert tensor.selected(plan).schedule.recompute
    with ResourceSession(
        plan,
        {
            "hf": lambda p: calculator.prepare_batch([atoms], resource_plan=p),
            "tensor": tensor.factory(compiler, cache),
        },
    ) as session:
        session.advance(0)
        native = session.provider("hf")
        hf_result = native.execute(strict=True).items[0]
        assert hf_result.energy == pytest.approx(
            Calculator().singlepoint(atoms).energy, abs=1e-10
        )
        feeds = {"x": np.linspace(-0.25, 0.75, 8192)}
        result = session.provider("tensor").execute(feeds)
        for i in range(6):
            assert result.outputs[f"r{i}"] == pytest.approx(
                (i + 2) * feeds["x"].sum(), rel=1e-11
            )
        actual = native.resource_diagnostics["observation"]["device_ledger"][
            "peak_bytes"
        ]
        actual += result.metrics["tracked_device_bytes"]
        assert actual <= plan.peak_bytes["device"]


def check(
    program: typing.Any,
    feeds: typing.Any,
    compiler: typing.Any,
    cache: typing.Any,
    schedule: typing.Any = None,
    **options: typing.Any,
) -> typing.Any:
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
            assert (
                result.metrics["observed_semantic_traffic_bytes"]
                == plan.semantic_traffic["total_bytes"]
            )
            assert (
                "not a hardware DRAM counter"
                in result.metrics["observed_traffic_scope"]
            )
        return result


def test_cub_block_reduce_matches_interpreter(
    compiler: typing.Any, cache: typing.Any
) -> None:
    i = Index("cub_rows", IndexSpace("cub_rows", "batch", 65))
    k = Index("cub_inner", IndexSpace("cub_inner", "batch", 4097))
    x = input_tensor("x", TensorSpec((i, k), role="input"))
    program = Program({"result": reduce_sum(x, (1,))})
    feeds = {"x": np.linspace(-0.25, 0.75, 65 * 4097).reshape(65, 4097)}
    result = check(
        program,
        feeds,
        compiler,
        cache,
        schedule=TensorSchedule(
            stream_reductions=True,
            reduction_provider="cub",
        ),
    )
    assert result.metrics["kernel_ms"] >= 0.0


def test_inplace_donation_executes_alias_safe_elementwise_chain(
    compiler: typing.Any, cache: typing.Any
) -> None:
    index = Index("i", IndexSpace("donation_axis", "batch", 4097))
    x = input_tensor("x", TensorSpec((index,), role="input"))
    transient = add(x, x, coefficients=(3, -1))
    result = multiply(transient, x)
    program = Program({"result": result})
    baseline = plan_cuda(program, compiler.target)
    schedule = TensorSchedule(inplace_donation=True, elements_per_thread=4)
    donated = plan_cuda(program, compiler.target, schedule=schedule)

    assert donated.arena_bytes < baseline.arena_bytes
    assert any(step.donated_from is not None for step in donated.steps)
    check(
        program,
        {"x": np.linspace(-1.25, 2.0, x.spec.size)},
        compiler,
        cache,
        schedule=schedule,
    )


@pytest.mark.parametrize("case", ["diagonal", "named_inputs", "inactive_operand"])
def test_generated_vjp_review_regressions_on_cuda(
    case: typing.Any, compiler: typing.Any, cache: typing.Any
) -> None:
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


def test_resource_aware_schedule_dimensions_execute_on_cuda(
    compiler: typing.Any, cache: typing.Any
) -> None:
    """Executable search axes keep exact numerical parity on the real backend."""
    i = Index("i", IndexSpace("rows", "batch", 17))
    j = Index("j", IndexSpace("columns", "batch", 19))
    k = Index("k", IndexSpace("inner", "batch", 33))
    rng = np.random.default_rng(508)

    x = input_tensor("x", TensorSpec((i, j), role="input"))
    check(
        Program({"out": add(multiply(x, x), x)}),
        {"x": rng.uniform(-0.5, 0.5, x.spec.shape)},
        compiler,
        cache,
        TensorSchedule(elements_per_thread=4),
    )

    r = input_tensor("r", TensorSpec((i, k), role="input"))
    check(
        Program({"out": reduce_sum(r, (1,))}),
        {"r": rng.uniform(-0.5, 0.5, r.spec.shape)},
        compiler,
        cache,
        TensorSchedule(elements_per_thread=2, reduction_unroll=4),
    )

    a = input_tensor("a", TensorSpec((i, k), role="input"))
    b = input_tensor("b", TensorSpec((k, j), role="input"))
    check(
        Program({"out": einsum("ik,kj->ji", a, b)}),
        {
            "a": rng.uniform(-0.5, 0.5, a.spec.shape),
            "b": rng.uniform(-0.5, 0.5, b.spec.shape),
        },
        compiler,
        cache,
        TensorSchedule(
            tile_m=11,
            tile_n=13,
            tile_k=17,
            direct_gemm=False,
            staging_width=4,
        ),
        library_bytes=0,
    )


@pytest.mark.parametrize("case", example_cases(), ids=lambda c: c.name)
def test_examples_and_every_intermediate(
    case: typing.Any, compiler: typing.Any, cache: typing.Any
) -> None:
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
def test_einsum_layouts_and_partial_tiles(
    expression: typing.Any,
    dimensions: typing.Any,
    compiler: typing.Any,
    cache: typing.Any,
) -> None:
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


def test_views_gathers_general_reductions_and_empty_axes(
    compiler: typing.Any, cache: typing.Any
) -> None:
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


def test_errors_recovery_detachment_and_independent_contexts(
    compiler: typing.Any, cache: typing.Any
) -> None:
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


def test_nonfinite_intermediate_and_minimum_budget(
    compiler: typing.Any, cache: typing.Any
) -> None:
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


def test_shape_buckets_budget_and_concurrent_system_independence(
    compiler: typing.Any, cache: typing.Any
) -> None:
    from vibeqc_compiler.tensor.cuda_batch import PreparedTensorBatch

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


def test_architecture_mismatch_is_explicit_before_execution(
    compiler: typing.Any, cache: typing.Any
) -> None:
    other = replace(
        compiler,
        target=cuda_target_info(
            "sm_80" if compiler.target.architecture != "sm_80" else "sm_86"
        ),
    )
    plan = plan_cuda(Program({"scalar": constant(3)}), other.target)
    with pytest.raises(ValueError, match="device architecture mismatch"):
        PreparedCuda(plan, compile_cuda(plan, other, cache))


def test_provider_allowance_guard_releases_a_rejected_handle(
    compiler: typing.Any, cache: typing.Any
) -> None:
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


@pytest.mark.parametrize("direct_gemm", [True, False])
def test_graph_replay_refresh_invalidation_and_profiling(
    compiler: typing.Any, cache: typing.Any, direct_gemm: typing.Any
) -> None:
    """Real capture for direct cuBLAS and packed partial-tile programs."""
    axis = IndexSpace("graph_axis", "batch", 17)
    i, j, k = (Index(name, axis) for name in "ijk")
    a = input_tensor("a", TensorSpec((i, j), role="input"))
    b = input_tensor("b", TensorSpec((j, k), role="input"))
    product = einsum("ij,jk->ik", a, b)
    program = Program({"product": product, "sum": reduce_sum(product, (0, 1))})
    plan = plan_cuda(
        program,
        compiler.target,
        schedule=TensorSchedule(
            direct_gemm=direct_gemm,
            tile_m=8,
            tile_n=8,
            tile_k=8,
        ),
    )
    artifact = compile_cuda(plan, compiler, cache)
    with PreparedCuda(plan, artifact, execution_mode="cuda-graph") as graph:
        for iteration in range(6):
            # Strided caller inputs and changing addresses must NOT invalidate
            # a graph that refers only to the owner's fixed device storage.
            values = np.arange(17 * 34, dtype=np.float64).reshape(17, 34)
            values /= 100
            values += iteration
            feeds = {
                "a": values[:, ::2],
                "b": np.eye(17) * (iteration + 1),
            }
            result = graph.execute(feeds)
            np.testing.assert_allclose(
                result.outputs["product"],
                feeds["a"] @ feeds["b"],
                rtol=2e-12,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                result.outputs["sum"], np.sum(feeds["a"] @ feeds["b"]), rtol=2e-12
            )
            assert result.metrics["graph_captures"] == (0 if iteration == 0 else 1)
            assert result.metrics["graph_replays"] == iteration
            assert result.metrics["graph_mode"] == (
                "warmup"
                if iteration == 0
                else "captured"
                if iteration == 1
                else "replay"
            )
        profiled = graph.execute(feeds, profile=True)
        assert profiled.metrics["graph_mode"] == "profiling"
        assert profiled.metrics["graph_captures"] == 1
        assert graph.execute(feeds).metrics["graph_captures"] == 1
        graph.invalidate_graph()
        assert graph.execute(feeds).metrics["graph_mode"] == "warmup"
        recaptured = graph.execute(feeds)
        assert recaptured.metrics["graph_captures"] == 2
        assert recaptured.metrics["graph_capture_attempts"] == 2
        assert (
            recaptured.metrics["owned_device_bytes"]
            == result.metrics["owned_device_bytes"]
        )
        assert recaptured.backend == "cuda-fp64-graph-replay"
    with pytest.raises(RuntimeError, match="closed"):
        graph.invalidate_graph()


def test_graph_arithmetic_failure_is_preserved_and_next_replay_recovers(
    compiler: typing.Any, cache: typing.Any
) -> None:
    i = Index("i", IndexSpace("axis", "batch", 8))
    x = input_tensor("x", TensorSpec((i,), role="input"))
    y = input_tensor("y", TensorSpec((i,), role="input"))
    plan = plan_cuda(Program({"ratio": divide(x, y)}), compiler.target)
    artifact = compile_cuda(plan, compiler, cache)
    good = {"x": np.arange(8, dtype=np.float64), "y": np.ones(8)}
    with PreparedCuda(plan, artifact, execution_mode="cuda-graph") as graph:
        graph.execute(good)
        graph.execute(good)
        with pytest.raises(RuntimeError, match="division by zero"):
            graph.execute({**good, "y": np.zeros(8)})
        with pytest.raises(RuntimeError, match="non-finite tensor"):
            graph.execute({"x": np.full(8, 1e308), "y": np.full(8, 1e-308)})
        recovered = graph.execute(good)
        np.testing.assert_array_equal(recovered.outputs["ratio"], good["x"])
        assert recovered.metrics["graph_captures"] == 1
        assert recovered.metrics["graph_mode"] == "replay"
        with pytest.raises(ValueError, match="non-finite tensor input"):
            graph.execute({**good, "x": np.full(8, np.nan)})
        assert graph.execute(good).metrics["graph_captures"] == 1


def test_graph_global_budget_falls_back_without_untracked_graph_storage(
    compiler: typing.Any, cache: typing.Any
) -> None:
    from vibeqc.resources import ResourceBudget, plan_resources
    from vibeqc_compiler.tensor.resources import tensor_resource_choices

    program = Program({"scalar": constant(3)})
    choices = tensor_resource_choices(program, compiler.target)
    resources = plan_resources([choices.request], ResourceBudget())
    plan = choices.selected(resources)
    artifact = compile_cuda(plan, compiler, cache)
    with PreparedCuda(
        plan,
        artifact,
        resource_plan=resources,
        resource_owner=choices.request.name,
        execution_mode="cuda-graph",
    ) as graph:
        for _ in range(3):
            result = graph.execute({})
            assert result.outputs["scalar"] == 3
            assert result.metrics["graph_mode"] == "fallback"
            assert result.metrics["graph_capture_attempts"] == 0
            assert "global resource plan" in result.metrics["graph_reason"]
            assert result.metrics["graph_retained_device_bytes"] == 0


def test_graph_fp32_falls_back_without_relabeling_precision(
    compiler: typing.Any, cache: typing.Any
) -> None:
    i = Index("i", IndexSpace("axis", "batch", 4))
    x = input_tensor("x", TensorSpec((i,), dtype="float32", role="input"))
    plan = plan_cuda(Program({"out": add(x, x)}), compiler.target)
    values = np.arange(4, dtype=np.float32)
    with PreparedCuda(
        plan, compile_cuda(plan, compiler, cache), execution_mode="cuda-graph"
    ) as prepared:
        for _ in range(3):
            result = prepared.execute({"x": values})
            assert result.backend == "cuda-fp32-ordinary-stream"
            assert result.metrics["precision"] == "fp32"
            np.testing.assert_array_equal(result.outputs["out"], 2 * values)


@pytest.mark.parametrize("family", ["elementwise", "contraction"])
def test_staged_tuning_qualifies_fresh_complete_cuda_endpoints(
    compiler: typing.Any, cache: typing.Any, family: typing.Any
) -> None:
    """Real CUDA screens may rank, but only fresh all-input gates may promote."""
    from vibeqc_compiler.tensor.cuda_search import TensorScreeningPolicy
    from vibeqc_compiler.tensor.cuda_tune import tune_cuda

    i = Index("i", IndexSpace("rows", "batch", 65))
    j = Index("j", IndexSpace("columns", "batch", 47))
    k = Index("k", IndexSpace("inner", "batch", 33))
    rng = np.random.default_rng(508)
    if family == "elementwise":
        x = input_tensor("x", TensorSpec((i, j), role="input"))
        program = Program({"out": add(multiply(x, x), x)})
        feeds = {"x": rng.uniform(-0.5, 0.5, x.spec.shape)}

        def oracle(values: typing.Any) -> typing.Any:
            return values["x"] * values["x"] + values["x"]
    else:
        x = input_tensor("x", TensorSpec((i, k), role="input"))
        y = input_tensor("y", TensorSpec((k, j), role="input"))
        program = Program({"out": einsum("ik,kj->ji", x, y)})
        feeds = {
            "x": rng.uniform(-0.5, 0.5, x.spec.shape),
            "y": rng.uniform(-0.5, 0.5, y.spec.shape),
        }

        def oracle(values: typing.Any) -> typing.Any:
            return np.einsum("ik,kj->ji", values["x"], values["y"])

    fixtures = [
        feeds,
        {name: np.asfortranarray(value * 0.7) for name, value in feeds.items()},
        {name: value[::-1] for name, value in feeds.items()},
    ]
    baseline = plan_cuda(program, compiler.target)
    selection = tune_cuda(
        baseline,
        compiler,
        fixtures,
        cache,
        schedules=[TensorSchedule(threads=t) for t in (64, 256, 512)],
        screening=TensorScreeningPolicy(maximum_finalists=1),
        repeats=5,
        maximum_seconds=180,
    )
    summary = selection.evidence["search_summary"]
    assert selection.evidence["screening_plan"]["active"]
    assert summary["screened_candidates"] == 3
    assert summary["endpoint_candidates"] == 1
    assert summary["shortlist_pruned"] == 2
    finalist = next(
        row for row in selection.evidence["candidates"] if row.get("endpoint_attempted")
    )
    assert len(finalist["samples"]) == len(fixtures)
    assert all(len(pairs) == 10 for pairs in finalist["samples"])
    assert finalist["status"] in ("accepted", "rejected")
    assert len(finalist["gates"]) == len(finalist["shared_gates"]) == len(fixtures)
    if selection.plan != baseline:
        assert finalist["status"] == "accepted"
        assert all(gate["passed"] for gate in finalist["gates"])
        assert all(gate["status"] == "pass" for gate in finalist["shared_gates"])
    else:
        assert not selection.evidence["selected_profiles"]
    with PreparedCuda(selection.plan, selection.artifact) as prepared:
        for values in fixtures:
            result = prepared.execute(values)
            np.testing.assert_allclose(
                result.outputs["out"], oracle(values), atol=1e-11, rtol=1e-10
            )
