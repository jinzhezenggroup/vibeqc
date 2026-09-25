"""Integer control tensors must not require floating-point precision records."""

import typing
from types import SimpleNamespace

import pytest
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.tensor.cuda_plan import plan_cuda
from vibeqc_compiler.tensor.cuda_search import (
    _fp64_accumulation_terms,
    estimate_schedule,
)
from vibeqc_compiler.tensor.ir import input_tensor, reduce_sum, runtime_indexed_select
from vibeqc_compiler.tensor.program import Program
from vibeqc_compiler.tensor.types import Index, IndexSpace, TensorSpec


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("count", (0, 3, 17))
def test_runtime_index_maps_have_zero_floating_accumulation_cost(
    dtype: str, count: int
) -> None:
    source_index = Index("i", IndexSpace("source", "batch", 5))
    selected_index = Index("q", IndexSpace("selected", "batch", count))
    source = input_tensor(
        "source", TensorSpec((source_index,), dtype=dtype, role="input")
    )
    mapping = input_tensor(
        "map", TensorSpec((selected_index,), dtype="int64", role="input")
    )
    selected = runtime_indexed_select(source, ((0, mapping),), selected_index)
    program = Program({"selected": selected})
    plan = plan_cuda(program, cuda_target_info("sm_80"), max_bytes=16 << 20)
    identity = plan.identity
    assert mapping not in plan.precision_by_node
    estimate = estimate_schedule(plan)
    assert estimate["estimated_fp64_accumulation_terms"] == 0
    assert estimate["estimated_effective_flops"] == plan.estimated_flops == 0
    assert estimate["profitability"]["static"]["arithmetic_operation_count"] == 0
    assert plan.identity == identity


def test_integer_skip_keeps_mixed_float_reduction_accounting() -> None:
    index = Index("q", IndexSpace("selected", "batch", 7))
    mapping = input_tensor("map", TensorSpec((index,), dtype="int64", role="input"))
    source = input_tensor("source", TensorSpec((index,), dtype="float32", role="input"))
    reduced = reduce_sum(source, (0,))
    plan: typing.Any = SimpleNamespace(
        steps=tuple(SimpleNamespace(node=node) for node in (mapping, source, reduced)),
        precision_by_node={
            source: SimpleNamespace(
                compute_dtype="float32", accumulation_dtype="float32"
            ),
            reduced: SimpleNamespace(
                compute_dtype="float32", accumulation_dtype="float64"
            ),
        },
    )
    assert _fp64_accumulation_terms(plan) == 7


def test_missing_float_precision_still_fails_closed() -> None:
    source = input_tensor("source", TensorSpec((), role="input"))
    plan: typing.Any = SimpleNamespace(
        steps=(SimpleNamespace(node=source),), precision_by_node={}
    )
    with pytest.raises(KeyError):
        _fp64_accumulation_terms(plan)
