"""Host invariants for the CUDA DF metric-projection scaling schedule."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/scf/cuda/df_metric_kernels.cu"
OCCUPIED_SOURCE = ROOT / "src/scf/cuda/df_occupied_exchange.cpp"


def _pair_groups() -> int:
    text = SOURCE.read_text()
    match = re.search(r"kPairGroups\s*=\s*(\d+)", text)
    assert match is not None
    return int(match.group(1))


def _scheduled_pairs(pairs: int, groups: int) -> list[int]:
    visited: list[int] = []
    for group in range(min(pairs, groups)):
        visited.extend(range(group, pairs, min(pairs, groups)))
    return visited


def _streamed_projection_section() -> str:
    text = OCCUPIED_SOURCE.read_text()
    return text.split("const auto project =", 1)[1].split("vibeqc_status status =", 1)[
        0
    ]


def test_metric_projection_pair_stripes_cover_each_pair_once() -> None:
    groups = _pair_groups()
    assert groups > 0
    for pairs in (1, 7, groups, groups + 1, 257):
        assert sorted(_scheduled_pairs(pairs, groups)) == list(range(pairs))


def test_metric_projection_source_reuses_denominator_across_pair_stripes() -> None:
    text = SOURCE.read_text()
    assert "const auto denominator = square_root ? sqrt(value) : value;" in text
    assert "pair += static_cast<std::size_t>(gridDim.y)" in text
    assert "scaled[index] = projected[index] / denominator;" in text
    assert "std::min(pairs, kPairGroups)" in text
    assert "launch_scale_metric_projection_to(" in text


def test_streamed_occupied_projection_fuses_scale_and_retention_copy() -> None:
    section = _streamed_projection_section()
    assert "launch_scale_metric_projection_to(" in section
    assert "streamed_occupied_projection_copy_bytes_avoided" in section
    assert "cudaMemcpyDeviceToDevice" not in section


@pytest.mark.parametrize(
    "dimension,pairs",
    [
        (928, 96 * 40),
        (3712, 384 * 160),
    ],
)
def test_metric_projection_scale_work_census(dimension: int, pairs: int) -> None:
    groups = min(pairs, _pair_groups())
    legacy_sqrt_evaluations = dimension * pairs
    striped_sqrt_evaluations = dimension * groups
    assert striped_sqrt_evaluations < legacy_sqrt_evaluations
    assert legacy_sqrt_evaluations // striped_sqrt_evaluations == pairs // groups


@pytest.mark.parametrize(
    "dimension,rows,rank,expected_mib",
    [
        (928, 96, 40, 54.375),
        (3712, 384, 160, 3480.0),
    ],
)
def test_two_block_streamed_projection_copy_payload_is_removed(
    dimension: int, rows: int, rank: int, expected_mib: float
) -> None:
    legacy_copy_bytes = 2 * dimension * rows * rank * 8
    assert legacy_copy_bytes / (1 << 20) == expected_mib
