"""Bounded scatter planning; materialization belongs to accepted source emission."""

from unittest.mock import patch

import pytest
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    batch_schedule,
    cuda_plan,
    input_tensor,
    scatter_add,
)
from vibeqc_compiler.tensor.batch_schedule import index_table_values


def _scatter_program(targets: int) -> Program:
    source = Index("i", IndexSpace("sources", "pair", 2))
    target = Index("j", IndexSpace("targets", "atom", targets))
    value = input_tensor("value", TensorSpec((source,), role="input"))
    return Program({"result": scatter_add(value, 0, (0, 0), target)})


@pytest.mark.parametrize("targets", [128, 2**40])
def test_planning_rejects_without_constructing_index_payload(targets: int) -> None:
    program = _scatter_program(targets)
    with (
        patch.object(
            batch_schedule,
            "scatter_add_inverted_table",
            side_effect=AssertionError("payload constructed during planning"),
        ),
        pytest.raises(ValueError, match="budget"),
    ):
        cuda_plan.plan_cuda(program, cuda_target_info("sm_80"), max_bytes=256)


def test_exact_bound_and_metrics_do_not_rebuild_payload() -> None:
    program = _scatter_program(9)
    target = cuda_target_info("sm_80")
    baseline = cuda_plan.plan_cuda(program, target)
    expected = (0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1)
    assert index_table_values(program.outputs["result"]) == expected
    with patch.object(
        batch_schedule,
        "scatter_add_inverted_table",
        side_effect=AssertionError("payload reconstructed during planning"),
    ):
        exact = cuda_plan.plan_cuda(program, target, max_bytes=baseline.peak_bytes)
        assert exact.index_table_bytes == 256
        assert exact.index_tables == baseline.index_tables
        assert exact.arena_bytes == baseline.arena_bytes
        with pytest.raises(ValueError, match="budget"):
            cuda_plan.plan_cuda(program, target, max_bytes=baseline.peak_bytes - 1)
