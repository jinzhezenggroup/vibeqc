from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "src/scf/cuda/df_metric_kernels.cu").read_text(encoding="utf-8")


def _decode_tile(pair: int, tiles: int) -> tuple[int, int]:
    low, high = 0, tiles
    while low + 1 < high:
        mid = (low + high) // 2
        if mid * (mid + 1) // 2 <= pair:
            low = mid
        else:
            high = mid
    return pair - low * (low + 1) // 2, low


def test_metric_symmetry_uses_compact_upper_tile_domain() -> None:
    kernel = SOURCE.split("__global__ void scale_eigenvectors_kernel", maxsplit=1)[0]
    assert "__shared__ std::size_t tile_row, tile_column;" in kernel
    assert "tile_row = pair - low * (low + 1) / 2;" in kernel
    assert "tile_column = low;" in kernel
    assert "const auto tile_pairs = tiles * (tiles + 1) / 2;" in SOURCE
    assert "dim3(static_cast<unsigned>(tile_pairs), 1, grid.z)" in SOURCE
    assert "blockIdx.y" not in kernel


def test_metric_symmetry_tile_decoder_covers_each_upper_tile_once() -> None:
    for tiles in (1, 2, 3, 7, 58, 232):
        actual = [_decode_tile(pair, tiles) for pair in range(tiles * (tiles + 1) // 2)]
        expected = [(row, column) for column in range(tiles) for row in range(column + 1)]
        assert actual == expected


def test_metric_symmetry_reduces_representative_launch_work() -> None:
    # These auxiliary dimensions match the representative streamed-DF shapes
    # used by the #1078 performance campaign. Counts are per system.
    for naux, square, triangle in ((928, 3364, 1711), (3712, 53824, 27028)):
        tiles = (naux + 15) // 16
        assert tiles * tiles == square
        assert tiles * (tiles + 1) // 2 == triangle
        assert triangle < square
