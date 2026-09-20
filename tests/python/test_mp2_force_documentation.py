"""Current-state documentation claims for conventional public MP2 forces."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_mp2_documentation_states_exact_force_and_batch_boundary() -> None:
    text = (ROOT / "docs/mp2.md").read_text(encoding="utf-8")
    for claim in (
        "force = -gradient",
        "Hartree/Bohr",
        "conventional canonical RHF-MP2",
        "CPU and CUDA",
        "RI-MP2 forces remain unsupported",
        "per-item",
        "measured_endpoint_peak_bytes",
        "measured_response_workspace_peak_bytes",
        "response_workspace_allocation_count",
        "issue193-conventional-force-b2",
    ):
        assert claim in text
    assert 'singlepoint(..., properties=("energy", "forces"))` rejects MP2' not in text


def test_method_table_no_longer_calls_conventional_mp2_forces_planned() -> None:
    text = (ROOT / "docs/methods.md").read_text(encoding="utf-8")
    assert "Conventional energy and analytic forces implemented on CPU/CUDA" in text
    assert (
        "RI energy implemented on CPU/CUDA; RI analytic forces remain C2 work" in text
    )
