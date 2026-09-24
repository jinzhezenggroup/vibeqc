"""Cooperative CUDA DF DIIS computes each symmetric Gram pair once."""

from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = (_REPOSITORY_ROOT / "src/scf/cuda/scf_diis_kernels.cu").read_text(
    encoding="utf-8"
)


def test_partial_launch_uses_unique_history_pairs_and_mirrors_results() -> None:
    assert "const auto pair_blocks = history * (history + 1U) / 2U;" in _SOURCE
    assert "dim3(parts, pair_blocks, batch_size)" in _SOURCE
    assert "decode_symmetric_pair(static_cast<std::uint32_t>(blockIdx.y), history" in _SOURCE
    assert "partials[forward] = value;" in _SOURCE
    assert "if (row != column)" in _SOURCE
    assert "partials[reverse] = value;" in _SOURCE


def test_cooperative_update_reuses_symmetric_gram_entries() -> None:
    assert (
        "static_cast<std::size_t>(count) * (count + 1U) / 2U" in _SOURCE
    )
    assert "decode_symmetric_pair(static_cast<std::uint32_t>(pair), count" in _SOURCE
    assert "matrix[static_cast<std::size_t>(column) * dimension + row] = dot;" in _SOURCE
    # The separate non-cooperative instantiation retains the historical square ordering.
    assert "static_cast<std::size_t>(count) * count" in _SOURCE
    assert "pair / count" in _SOURCE
    assert "pair % count" in _SOURCE


def test_full_history_work_census_for_retained_df_shapes() -> None:
    history = 8
    square_pairs = history * history
    unique_pairs = history * (history + 1) // 2
    assert (square_pairs, unique_pairs) == (64, 36)

    expected = {
        384: (9_437_184, 5_308_416, 144, 81),
        768: (37_748_736, 21_233_664, 576, 324),
    }
    for nbf, (old_products, new_products, old_mib, new_mib) in expected.items():
        vector_size = nbf * nbf
        assert square_pairs * vector_size == old_products
        assert unique_pairs * vector_size == new_products
        # Each residual dot reads two FP64 input values per scalar product.
        assert old_products * 16 // (1024 * 1024) == old_mib
        assert new_products * 16 // (1024 * 1024) == new_mib
        assert new_products * 16 < old_products * 16
