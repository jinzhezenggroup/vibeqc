"""Static index storage is admitted without materializing compiler tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from vibeqc_compiler.tensor.ir import Node


def scatter(source_count: int, target_count: int, dtype: str = "float64") -> Node:
    source = Index("p", IndexSpace("pairs", "pair", source_count))
    target = Index("a", IndexSpace("atoms", "atom", target_count))
    data = input_tensor("values", TensorSpec((source,), dtype=dtype, role="input"))
    positions = tuple(i % target_count for i in range(source_count))
    return scatter_add(data, 0, positions, target)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("fuse", [False, True])
def test_infeasible_sparse_scatter_rejects_before_building_index_payload(
    dtype: str, fuse: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    program = Program({"out": scatter(1, 1 << 40, dtype)})

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("index payload materialized before resource admission")

    monkeypatch.setattr(cuda_plan, "_index_table_values", forbidden)
    with pytest.raises(ValueError, match="byte budget"):
        cuda_plan.plan_cuda(
            program,
            cuda_target_info("sm_120"),
            max_bytes=1 << 20,
            schedule=cuda_plan.TensorSchedule(fuse=fuse),
        )


def test_plan_and_repeated_metrics_do_not_rebuild_static_index_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = scatter(7, 5)
    values = cuda_plan._index_table_values(node)
    assert values is not None
    expected = cuda_plan.aligned(len(values) * 8)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("metadata query rebuilt immutable index payload")

    monkeypatch.setattr(cuda_plan, "_index_table_values", forbidden)
    plan = cuda_plan.plan_cuda(Program({"out": node}), cuda_target_info("sm_120"))
    identity = plan.identity
    for _ in range(3):
        assert plan.index_table_bytes == expected
        assert plan.ragged_resources["index_table_bytes"] == expected
        assert plan.identity == identity
