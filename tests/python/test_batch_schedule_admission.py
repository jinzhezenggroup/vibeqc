"""Batch schedules count topology storage without materializing it in admission."""

import pytest
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    cuda_plan,
    input_tensor,
    scatter_add,
)


@pytest.mark.parametrize("target_count,budget", [(100_000_000, 1024), (3, 1 << 20)])
def test_scatter_table_is_not_built_by_planning_or_metrics(
    monkeypatch: pytest.MonkeyPatch, target_count: int, budget: int
) -> None:
    source = Index("s", IndexSpace("sources", "batch", 2))
    target = Index("t", IndexSpace("targets", "batch", target_count))
    x = input_tensor("x", TensorSpec((source,), role="input"))
    program = Program({"out": scatter_add(x, 0, (0, 1), target)})

    def forbidden(node: object) -> None:
        pytest.fail("materialized index payload before its admission")

    monkeypatch.setattr(cuda_plan, "_index_table_values", forbidden)
    if target_count > 3:
        with pytest.raises(ValueError, match="budget"):
            cuda_plan.plan_cuda(program, cuda_target_info("sm_120"), max_bytes=budget)
    else:
        plan = cuda_plan.plan_cuda(
            program, cuda_target_info("sm_120"), max_bytes=budget
        )
        assert plan.index_table_bytes == cuda_plan.aligned((target_count + 1 + 2) * 8)
