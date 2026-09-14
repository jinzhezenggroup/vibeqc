"""Partitioned storage must preserve scientific values and reject corrupt parts."""

import json
from pathlib import Path

import pytest

from tools.vibeqc_validation.record import decode_record, load_record
from tools.vibeqc_validation.retention import digest


def test_migrated_records_match_original_scientific_values():
    """Compare all reconstructed values with hashes taken before migration."""
    root = Path(__file__).resolve().parents[2]
    audit = json.loads(
        (root / "benchmarks/results/retention-size-limit/migration.json").read_text()
    )
    for entry in audit["records"]:
        restored = load_record(root / entry["path"])
        canonical = json.dumps(
            restored, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        assert digest(canonical) == entry["canonical_sha256"]


def record(part):
    return {
        "schema": "example",
        "record_parts": {
            "timings": [
                {"path": "water.json", "bytes": len(part), "sha256": digest(part)}
            ]
        },
    }


def test_parts_restore_order_and_float_precision(tmp_path):
    values = [{"seconds": 0.12345678901234567}, {"seconds": 1e-14}]
    part = json.dumps(values).encode()
    main = json.dumps(record(part)).encode()
    assert decode_record(main, {"water.json": part}) == {
        "schema": "example",
        "timings": values,
    }
    (tmp_path / "evidence.json").write_bytes(main)
    (tmp_path / "water.json").write_bytes(part)
    assert load_record(tmp_path / "evidence.json")["timings"] == values


@pytest.mark.parametrize(
    "damage", ["bytes", "missing", "overwrite", "duplicate", "traversal", "type"]
)
def test_corrupt_parts_fail(damage):
    part = b"[1.25, 0.5]"
    main = record(part)
    files = {"water.json": part}
    entries = main["record_parts"]["timings"]
    if damage == "bytes":
        files["water.json"] += b" "
    elif damage == "missing":
        files.clear()
    elif damage == "overwrite":
        main["timings"] = []
    elif damage == "duplicate":
        entries *= 2
    elif damage == "traversal":
        entries[0]["path"] = "../water.json"
    else:
        part = b"{}"
        files["water.json"] = part
        entries[0].update(bytes=len(part), sha256=digest(part))
    with pytest.raises((ValueError, KeyError, TypeError)):
        decode_record(json.dumps(main).encode(), files)


def test_part_symlink_cannot_escape_directory(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"[]")
    directory = tmp_path / "publication"
    directory.mkdir()
    (directory / "water.json").symlink_to(outside)
    (directory / "evidence.json").write_text(json.dumps(record(b"[]")))
    with pytest.raises(ValueError, match="escapes"):
        load_record(directory / "evidence.json")
