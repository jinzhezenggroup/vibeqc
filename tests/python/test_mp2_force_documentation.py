"""Current-state documentation claims for conventional public MP2 forces."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_mp2_documentation_states_exact_force_and_batch_boundary() -> None:
    text = (ROOT / "docs/developer/mp2.md").read_text(encoding="utf-8")
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
    text = (ROOT / "docs/user/methods.md").read_text(encoding="utf-8")
    assert "[public method table](../public_methods.md)" in text
    table = (ROOT / "docs/public_methods.md").read_text(encoding="utf-8")
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in table.splitlines()
        if line.lstrip().startswith("|")
    ]
    mp2 = [row for row in rows if row[0] == "`mp2`"]
    assert len(mp2) == 1 and len(mp2[0]) == 6
    assert set(mp2[0][2].split(", ")) == {"`energy`", "`forces`"}
    assert mp2[0][3] == "yes" and mp2[0][5] == "available"
    contract = (ROOT / "docs/developer/mp2.md").read_text(encoding="utf-8")
    assert "CPU and CUDA" in contract
    assert "RI-MP2 forces remain unsupported" in contract
