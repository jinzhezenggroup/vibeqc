"""CUDA RI-MP2 residency/work-ledger regression for issue #367."""

import json
import os
from pathlib import Path

import pytest
from vibeqc import Calculator

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_MP2_CUDA_TEST") != "1",
    reason="requires explicitly allocated CUDA device and native library",
)


def test_cuda_ri_mp2_keeps_transformed_b_on_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = tmp_path / "ri-mp2.jsonl"
    monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, -1.432, 1.107)),
        ("H", (0.0, 1.432, 1.107)),
    ]
    result = Calculator(
        method="mp2", basis="def2-svp", device="cuda", density_fitting="cuda"
    ).singlepoint(atoms, properties=("energy",))
    assert result.correlation.mo_host_staging is False
    assert result.correlation.mo_transfer_bytes > 0

    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    ri = [row for row in rows if row["operation"] == "ri_mp2_energy"]
    assert len(ri) == 1
    record = ri[0]
    counters = record["counters"]
    assert counters["ri_mp2_virtual_blocks"] == 1
    assert counters["ri_mp2_source_passes"] == 1
    assert counters["ri_mp2_source_row_generations"] == record["nbf"]
    assert counters["ri_mp2_resident_b_bytes"] > 0
    assert counters["ri_mp2_final_d2h_bytes"] == 16
    assert counters["ri_mp2_transfer_bytes"] == (
        counters["ri_mp2_h2d_bytes"] + counters["ri_mp2_d2h_bytes"]
    )
    names = {region["name"] for region in record["regions"]}
    assert "ri_mp2_ao_to_mo" in names
    assert "ri_mp2_energy_contraction" in names
    assert "transformed_three_center_generation" in names
