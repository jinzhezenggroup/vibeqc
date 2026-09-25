"""Keep CUDA resource admission conservative for complete PTXAS stack reports."""

import pytest
from vibeqc_compiler.common.cuda_resources import parse_resources


def _record(frame: int, cumulative: int | None, newline: str = "\n") -> str:
    suffix = (
        "" if cumulative is None else f", {cumulative} bytes cumulative stack size"
    )
    return newline.join(
        (
            "ptxas info : Function properties for bulk_xc_census_probe",
            f"    {frame} bytes stack frame, 0 bytes spill stores, 0 bytes spill loads",
            f"ptxas info : Used 32 registers, used 0 barriers{suffix}, 0 bytes lmem",
            "",
        )
    )


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize(
    ("frame", "cumulative", "expected"),
    ((16, 256, 256), (256, 16, 256), (32, 32, 32), (0, None, 0)),
)
def test_stack_admission_uses_larger_reported_bound(
    newline: str, frame: int, cumulative: int | None, expected: int
) -> None:
    records = parse_resources(_record(frame, cumulative, newline))
    assert len(records) == 1
    record = records[0]
    assert record.stack_bytes == expected
    assert record.local_bytes == 0
    assert record.shared_bytes == 0
    assert record.spill_store_bytes == record.spill_load_bytes == 0


def test_cumulative_stack_stays_with_its_function() -> None:
    first = _record(16, 256).replace("bulk_xc_census_probe", "caller")
    second = _record(8, 64).replace("bulk_xc_census_probe", "callee")
    records = parse_resources(first + second)
    assert [(record.function, record.stack_bytes) for record in records] == [
        ("caller", 256),
        ("callee", 64),
    ]


def test_incomplete_diagnostic_does_not_invent_a_resource_record() -> None:
    assert (
        parse_resources("ptxas info : Used 32 registers, 256 bytes cumulative stack size")
        == ()
    )
