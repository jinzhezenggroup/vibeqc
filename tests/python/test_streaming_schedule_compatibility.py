"""Legacy positional schedules must not implicitly opt into streaming."""

import pytest
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule
from vibeqc_compiler.tensor.cuda_search import TensorScheduleSpace


@pytest.mark.parametrize("direct", (False, True))
def test_legacy_schedule_positional_direct_gemm(direct: bool) -> None:
    schedule = TensorSchedule(128, 128, 128, 128, False, False, False, direct)
    assert schedule.direct_gemm is direct
    assert schedule.stream_reductions is False


@pytest.mark.parametrize("direct", ((False,), (True,)))
def test_legacy_search_positional_direct_gemm(direct: tuple[bool, ...]) -> None:
    space = TensorScheduleSpace((False,), (False,), (False,), direct)
    assert space.direct_gemm == direct
    assert space.stream_reductions == (False,)
    assert all(not schedule.stream_reductions for schedule in space.generate(4))
