"""Real-device parity of producer layouts; requires an allocated CUDA job."""

import os
import typing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_tensor_layout import gemm_producer_case, producer_case
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import execute, linearize, transpose_program
from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda
from vibeqc_compiler.tensor.cuda_resident import PreparedResident, compile_resident

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)


@pytest.fixture(scope="module")
def compiler() -> typing.Any:
    return CudaCompilerAdapter(
        Path(os.environ["VIBEQC_NVCC"]),
        cuda_target_info(os.environ.get("VIBEQC_TENSOR_ARCH", "sm_120")),
    )


@pytest.fixture(scope="module")
def cache(tmp_path_factory: typing.Any) -> typing.Any:
    return (
        Path(os.environ["VIBEQC_TENSOR_CACHE"])
        if "VIBEQC_TENSOR_CACHE" in os.environ
        else tmp_path_factory.mktemp("layout-cuda")
    )


def check(
    program: typing.Any,
    feeds: typing.Any,
    compiler: typing.Any,
    cache: typing.Any,
    *,
    schedule: typing.Any = None,
    expected: typing.Any = None,
) -> typing.Any:
    expected = execute(program, feeds).outputs if expected is None else expected
    plan = plan_cuda(
        program,
        compiler.target,
        schedule=TensorSchedule(layouts=True) if schedule is None else schedule,
    )
    artifact = compile_cuda(plan, compiler, cache)
    with PreparedCuda(plan, artifact) as prepared:
        for profile in (False, True):
            result = prepared.execute(feeds, profile=profile)
            for name, values in expected.items():
                np.testing.assert_allclose(
                    result.outputs[name], values, atol=1e-11, rtol=1e-10
                )
            assert result.metrics["owned_device_bytes"] == plan.allocation_bytes
            assert result.metrics["provider_retained_bytes"] <= plan.provider_bytes
            assert result.metrics["predicted_peak_bytes"] <= plan.max_bytes
    return plan


@pytest.mark.parametrize(
    "kind",
    [
        "multiply",
        "add",
        "divide",
        "transpose",
        "gather",
        "slice",
        "broadcast",
        "reshape",
        "reduce",
    ],
)
def test_generic_producers_with_physical_output_order(
    kind: typing.Any, compiler: typing.Any, cache: typing.Any
) -> None:
    program, feeds, _ = producer_case(kind)
    plan = check(program, feeds, compiler, cache)
    assert plan.layout_decision.changed_steps
    assert plan.panel_bytes == 0


@pytest.mark.parametrize(
    "packed_producer,packed_consumer", [(False, False), (True, False), (False, True)]
)
def test_direct_and_packed_gemm_read_write_alternate_layouts(
    packed_producer: typing.Any,
    packed_consumer: typing.Any,
    compiler: typing.Any,
    cache: typing.Any,
) -> None:
    program, feeds, _ = gemm_producer_case(
        packed_producer=packed_producer, packed_consumer=packed_consumer
    )
    assert check(program, feeds, compiler, cache).layout_decision.changed_steps


def test_both_operands_require_joint_producer_selection(
    compiler: typing.Any, cache: typing.Any
) -> None:
    program, feeds, _ = producer_case(both=True)
    plan = check(program, feeds, compiler, cache)
    assert len(plan.layout_decision.changed_steps) == 2


@pytest.mark.parametrize("caller_layout", ["fortran", "negative", "broadcast"])
def test_caller_strides_remain_a_staging_contract(
    caller_layout: typing.Any, compiler: typing.Any, cache: typing.Any
) -> None:
    program, feeds, _ = producer_case()
    transforms = {
        "fortran": np.asfortranarray,
        "negative": lambda a: a[..., ::-1],
        "broadcast": lambda a: np.broadcast_to(a[..., :1], a.shape),
    }
    check(
        program,
        {name: transforms[caller_layout](a) for name, a in feeds.items()},
        compiler,
        cache,
    )


@pytest.mark.parametrize(
    "axis,size", [(axis, size) for axis in "ibkj" for size in (0, 1)]
)
def test_empty_and_singleton_domains(
    axis: typing.Any, size: typing.Any, compiler: typing.Any, cache: typing.Any
) -> None:
    program, feeds, _ = producer_case(
        dimensions={"i": 3, "b": 2, "k": 7, "j": 5, axis: size}
    )
    check(program, feeds, compiler, cache)


@pytest.mark.parametrize(
    "views,fuse,recompute",
    [(True, False, False), (True, True, False), (True, True, True)],
)
def test_existing_view_fusion_recomputation_schedules(
    views: typing.Any,
    fuse: typing.Any,
    recompute: typing.Any,
    compiler: typing.Any,
    cache: typing.Any,
) -> None:
    program, feeds, _ = producer_case("transpose")
    check(
        program,
        feeds,
        compiler,
        cache,
        schedule=TensorSchedule(
            layouts=True, views=views, fuse=fuse, recompute=recompute
        ),
    )


@pytest.mark.parametrize("direction", ["jvp", "vjp"])
def test_generated_derivatives_match_independent_analytic_equations(
    direction: typing.Any, compiler: typing.Any, cache: typing.Any
) -> None:
    program, feeds, _ = producer_case()
    x, y = feeds["x"], feeds["y"]
    rng = np.random.default_rng(1509)
    if direction == "jvp":
        dx = rng.normal(size=x.shape)
        derivative = linearize(program, ["x"])
        values = {**feeds, "d_x": dx}
        expected = {"d_out": 2 * np.einsum("ibk,ibk,bkj->bij", x, dx, y)}
    else:
        bar = rng.normal(size=(2, 3, 5))
        derivative = transpose_program(program, ["out"], inputs=["x"])
        values = {**feeds, "bar_out": bar}
        expected = {"bar_x": 2 * x * np.einsum("bij,bkj->ibk", bar, y)}
    plan = check(derivative.program, values, compiler, cache, expected=expected)
    assert plan.layout_decision.changed_steps


def test_resident_abi_and_artifact_guards_keep_logical_outputs(
    compiler: typing.Any, cache: typing.Any
) -> None:
    program, feeds, _ = producer_case()
    plan = plan_cuda(program, compiler.target, schedule=TensorSchedule(layouts=True))
    expected = execute(program, feeds).outputs
    artifact = compile_resident(plan, compiler, cache)
    with PreparedResident(plan, artifact) as prepared:
        prepared.upload(feeds)
        for _ in range(2):
            outputs, metrics = prepared.run()
            assert metrics["h2d_bytes"] == 0
            np.testing.assert_allclose(
                prepared.download(outputs["out"]),
                expected["out"],
                atol=1e-11,
                rtol=1e-10,
            )
    baseline = plan_cuda(program, compiler.target)
    with pytest.raises(ValueError, match="plan identity mismatch"):
        PreparedCuda(baseline, artifact)


@pytest.mark.parametrize(
    "kind,error", [("multiply", "non-finite"), ("divide", "division by zero")]
)
def test_error_boundaries_and_recovery_after_layout_change(
    kind: typing.Any, error: typing.Any, compiler: typing.Any, cache: typing.Any
) -> None:
    program, feeds, _ = producer_case(kind)
    plan = plan_cuda(program, compiler.target, schedule=TensorSchedule(layouts=True))
    artifact = compile_cuda(plan, compiler, cache)
    with PreparedCuda(plan, artifact) as prepared:
        bad = {
            **feeds,
            "x": np.full_like(feeds["x"], 1e200 if kind == "multiply" else 0.0),
        }
        with pytest.raises(RuntimeError, match=error):
            prepared.execute(bad)
        np.testing.assert_allclose(
            prepared.execute(feeds).outputs["out"],
            execute(program, feeds).outputs["out"],
            atol=1e-11,
            rtol=1e-10,
        )


def test_budget_shrunk_packed_scatter_with_noncanonical_intermediate(
    compiler: typing.Any, cache: typing.Any
) -> None:
    program, feeds, _ = gemm_producer_case(packed_producer=True)
    plan = plan_cuda(
        program,
        compiler.target,
        schedule=TensorSchedule(layouts=True, tile_m=2, tile_n=3, tile_k=2),
    )
    # Force every partial panel and keep the alternate producer output layout.
    check(
        program,
        feeds,
        compiler,
        cache,
        schedule=replace(plan.schedule, tile_m=1, tile_n=1, tile_k=1),
    )


@pytest.mark.parametrize("kind", ["shared", "grouped"])
def test_shared_operands_and_grouped_label_orders(
    kind: typing.Any, compiler: typing.Any, cache: typing.Any
) -> None:
    from test_tensor_layout import grouped_output_case, self_product_case

    program, feeds, _ = (
        self_product_case() if kind == "shared" else grouped_output_case()
    )
    assert check(program, feeds, compiler, cache).layout_decision.changed_steps
