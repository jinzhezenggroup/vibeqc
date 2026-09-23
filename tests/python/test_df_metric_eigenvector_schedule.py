"""Host invariants for CUDA DF metric-eigenvector scaling."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/scf/cuda/df_metric_kernels.cu"


def _column_kernel(text: str) -> str:
    return text.split("__global__ void scale_eigenvectors_column_kernel", 1)[1].split(
        "/** Scale a bounded pair stripe", 1
    )[0]


def test_metric_eigenvector_scaling_uses_column_stripes() -> None:
    text = SOURCE.read_text()
    kernel = _column_kernel(text)
    assert "const std::size_t column = static_cast<std::size_t>(blockIdx.x);" in kernel
    assert "const std::size_t system = static_cast<std::size_t>(blockIdx.y);" in kernel
    assert "__shared__ double column_scale;" in kernel
    assert "if (threadIdx.x == 0) column_scale = scales[system * dimension + column];" in kernel
    assert "row += blockDim.x" in kernel
    assert "scaled_eigenvectors[element] = eigenvectors[element] * column_scale;" in kernel
    assert "element %" not in kernel
    assert "element /" not in kernel
    assert "scale_eigenvectors_flat_kernel<<<grid, block, shared_bytes, stream>>>" in text


@pytest.mark.parametrize(
    "dimension,legacy_blocks,column_blocks,legacy_index_divmods",
    [
        (928, 3_364, 928, 2_583_552),
        (3_712, 53_824, 3_712, 41_336_832),
    ],
)
def test_metric_eigenvector_column_schedule_work_census(
    dimension: int, legacy_blocks: int, column_blocks: int, legacy_index_divmods: int
) -> None:
    """Pin the 256-thread singleton plan-setup census used by #1078."""
    elements = dimension * dimension
    assert (elements + 255) // 256 == legacy_blocks
    assert dimension == column_blocks
    assert 3 * elements == legacy_index_divmods

    # One column scale is fetched globally per block instead of once per element.
    assert column_blocks < legacy_blocks
    assert dimension < elements
