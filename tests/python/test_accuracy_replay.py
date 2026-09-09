"""A recorded fixed-density audit must replay without executing SCF again."""

import json
from pathlib import Path

import pytest
from vibeqc.profiles import canonical_hash

from tools.validate_accuracy import run
from tools.vibeqc_numerics.replay import replay


@pytest.fixture
def report(tmp_path):
    run(tmp_path, names=["h2"])
    return tmp_path / "report.json"


def test_strict_operator_replay_and_corruption(report, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("an operator replay must not launch SCF")

    monkeypatch.setattr("tools.vibeqc_numerics.audit.probe_hf", forbidden)
    result = replay(report)
    assert result["passed"] and result["scf_executed"] is False
    assert sum(row["status"] == "pass" for row in result["rows"]) == 13
    archive = report.parent / "states.npz"
    archive.write_bytes(archive.read_bytes()[:-4])
    with pytest.raises(ValueError, match="size"):
        replay(report)


def test_replay_checks_report_identity_before_array_allocation(report):
    record = json.loads(report.read_text())
    record["cases"][0]["model"]["basis_hash"] = "altered"
    report.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checksum"):
        replay(report)


@pytest.mark.parametrize(
    "cases", [[], [{"name": "failed", "status": "strict_reference_failed"}]]
)
def test_replay_requires_successful_reference_evidence(report, cases):
    record = json.loads(report.read_text())
    record.pop("record_hash")
    record["cases"] = cases
    record["record_hash"] = canonical_hash(record)
    report.write_text(json.dumps(record))
    assert replay(report)["passed"] is False


def test_extra_holdouts_have_stable_independent_reference_generations():
    from tools.vibeqc_numerics.fixtures import REFERENCE_DIRECTORY, accuracy_suite

    assert {r["inputs"]["name"] for r in accuracy_suite()} >= {"hf", "h2-def2-svp"}
    rows = json.loads((Path(REFERENCE_DIRECTORY) / "stability.json").read_text())[
        "rows"
    ]
    assert len(rows) == 2
    assert all(row["first_generated_at"] != row["second_generated_at"] for row in rows)
    assert all(row["first_record_hash"] != row["second_record_hash"] for row in rows)
