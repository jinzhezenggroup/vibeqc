"""#149 A: plans on CPU and explicit opt-in real-device equation parity."""

import os
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import execute
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda

from tools.vibeqc_cc.cuda import PreparedRCCSDResidual, rccsd_program
from tools.vibeqc_cc.doubles import build_ccsd_program
from tools.vibeqc_cc.oracle import dense_feeds, random_case


@pytest.mark.parametrize("shape", [(1, 3), (2, 3)])
def test_trace_is_original_equation_and_budget_keeps_every_node(shape):
    original = build_ccsd_program(*shape)
    traced = rccsd_program(*shape, trace=True)
    assert len(traced.outputs) == len(original.outputs) + len(original.live_nodes)
    feeds = dense_feeds(*random_case(*shape, seed=149))
    reference = execute(original, feeds).outputs
    actual = execute(traced, feeds).outputs
    for name in original.outputs:
        np.testing.assert_array_equal(actual[name], reference[name])
    target = cuda_target_info("sm_90")
    unit = plan_cuda(
        traced,
        target,
        schedule=TensorSchedule(tile_m=1, tile_n=1, tile_k=1),
        library_bytes=0,
    )
    constrained = plan_cuda(traced, target, max_bytes=unit.peak_bytes, library_bytes=0)
    assert constrained.peak_bytes <= unit.peak_bytes
    with pytest.raises(ValueError, match="infeasible"):
        plan_cuda(traced, target, max_bytes=unit.peak_bytes - 1, library_bytes=0)


@pytest.mark.skipif(
    os.environ.get("VIBEQC_CC_CUDA_TEST") != "1",
    reason="requires explicitly allocated GPU validation window",
)
@pytest.mark.parametrize("shape", [(1, 3), (2, 3)])
def test_real_device_every_node_repeat_failure_and_two_contexts(shape, tmp_path):
    from vibeqc.profiles import find_nvcc

    nvcc = find_nvcc()
    assert nvcc is not None
    compiler = CudaCompilerAdapter(
        nvcc, cuda_target_info(os.environ["VIBEQC_TENSOR_ARCH"])
    )
    cache = Path(os.environ.get("VIBEQC_TENSOR_CACHE", tmp_path))
    feeds = dense_feeds(*random_case(*shape, seed=149))
    expected = execute(rccsd_program(*shape, trace=True), feeds).outputs
    other_feeds = dense_feeds(*random_case(*shape, seed=193))
    other_expected = execute(rccsd_program(*shape, trace=True), other_feeds).outputs
    with (
        PreparedRCCSDResidual(*shape, compiler, cache, trace=True) as first,
        PreparedRCCSDResidual(*shape, compiler, cache, trace=True) as second,
    ):
        before = first.execute(feeds)
        bad = {**feeds, "t1": np.full(shape, np.nan)}
        with pytest.raises(ValueError, match="non-finite"):
            first.execute(bad)
        for result, reference in (
            (before, expected),
            (second.execute(other_feeds), other_expected),
            (first.execute(feeds), expected),
        ):
            for name, ref in reference.items():
                np.testing.assert_allclose(
                    result.outputs[name],
                    ref,
                    atol=1e-11,
                    rtol=1e-10,
                    err_msg=name,
                )
            assert result.metrics["owned_device_bytes"] == first.plan.allocation_bytes
        assert not np.shares_memory(
            before.outputs["singles_residual"], expected["singles_residual"]
        )
    with pytest.raises(RuntimeError, match="closed"):
        first.execute(feeds)
