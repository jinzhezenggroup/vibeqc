"""Host invariants for the CUDA DF metric-projection scaling schedule."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/scf/cuda/df_metric_kernels.cu"


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


def test_metric_projection_pair_stripes_cover_each_pair_once() -> None:
    groups = _pair_groups()
    assert groups > 0
    for pairs in (1, 7, groups, groups + 1, 257):
        assert sorted(_scheduled_pairs(pairs, groups)) == list(range(pairs))


def test_metric_projection_source_reuses_denominator_across_pair_stripes() -> None:
    text = SOURCE.read_text()
    assert "const auto denominator = square_root ? sqrt(value) : value;" in text
    assert "pair += static_cast<std::size_t>(gridDim.y)" in text
    assert "projected[pair * dimension + direction] /= denominator;" in text
    assert "std::min(pairs, kPairGroups)" in text


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
