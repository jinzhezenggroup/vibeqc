"""Production packed-response selection must use general workload policy."""

from pathlib import Path


def test_packed_response_auto_has_no_benchmark_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/scf/cuda/df_gradient_bridge.cu").read_text()

    assert "df_packed_response_preferred(" in source
    assert "NVIDIA GeForce RTX 5090" not in source
    assert "n == 768 && a == 768" not in source
    assert "borrowed->occupied_factors[0].rank == 160" not in source
