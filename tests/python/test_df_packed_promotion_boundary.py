"""Unmeasured generalized packed-response policy is not a production default."""

from pathlib import Path


def test_generalized_packed_profile_remains_candidate_only() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/scf/cuda/df_gradient_bridge.cu").read_text()
    assert "df_packed_response_preferred(" not in source
    # Keep the previously qualified route rather than disabling all packed work.
    assert 'std::string_view(properties.name) == "NVIDIA GeForce RTX 5090"' in source
    assert "n == 768 && a == 768" in source
    assert "borrowed->occupied_factors[0].rank == 160" in source
