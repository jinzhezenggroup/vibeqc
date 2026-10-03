"""Storage-only PBE0 packing preserves every complete and failed observation."""

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "benchmarks/results/pbe0-def2-svp-20261003"
ORIGINALS = {
    *(f"water{atoms}.json" for atoms in (3, 6, 12, 24, 48)),
    *(f"control-water{atoms}.json" for atoms in (3, 12, 24, 48)),
    "failed-integration-water96.json",
    "failed-preintegration-water96.json",
}


def test_complete_control_and_failed_reports_keep_exact_original_bytes() -> None:
    manifest = json.loads((DIRECTORY / "report-storage.json").read_text())
    assert manifest["schema"] == "generativeqc.pbe0-lossless-report-storage.v1"
    assert manifest["original_revision"] == "ab9a8490970d2b1dfaf711398b9069c5318ecb62"
    members = manifest["members"]
    assert len(members) == len(ORIGINALS)
    assert {member["original"] for member in members} == ORIGINALS
    for member in members:
        assert member["stored"] == member["original"] + ".gz"
        assert not (DIRECTORY / member["original"]).exists()
        stored = (DIRECTORY / member["stored"]).read_bytes()
        assert len(stored) == member["stored_bytes"]
        assert hashlib.sha256(stored).hexdigest() == member["stored_sha256"]
        original = gzip.decompress(stored)
        assert len(original) == member["original_bytes"]
        assert hashlib.sha256(original).hexdigest() == member["original_sha256"]
        report = json.loads(original)
        assert report["atoms"] in (3, 6, 12, 24, 48, 96)
        assert set(report) == {"atoms", "engines", "protocol", "independent_references"}
    assert (
        sum(member["original_bytes"] - member["stored_bytes"] for member in members)
        == 614_539
    )
