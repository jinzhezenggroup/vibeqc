"""Retire benchmark identities without promoting an unqualified default."""

import re
from pathlib import Path


def test_packed_response_auto_has_no_benchmark_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/scf/cuda/df_gradient_bridge.cu").read_text()
    assert "NVIDIA GeForce RTX 5090" not in source
    assert "n == 768 && a == 768" not in source
    assert "borrowed->occupied_factors[0].rank == 160" not in source
    assert "df_packed_response_preferred(" not in source
    assert "packed_default" not in source


def test_packed_response_requires_explicit_opt_in_and_ownership() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/scf/cuda/df_gradient_bridge.cu").read_text()
    match = re.search(r"const bool packed_pairs\s*=([^;]+);", source)
    assert match is not None
    expression = " ".join(match.group(1).split())
    assert expression == (
        'pair_policy == "packed" && shell_execution && '
        "full_shell_domain && borrowed && borrowed->occupied_response"
    )
    # Automatic qualified shell execution still falls back to symmetric pairs.
    assert '(pair_policy == "auto" && promoted_default)' in source
