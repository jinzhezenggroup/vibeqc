"""The archived scientific audit must fail closed and recompute derived claims."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

EVIDENCE = Path(__file__).resolve().parents[2] / "benchmarks/results/issue206-post418"
SCRIPT = EVIDENCE / "reproduction/analyze.py"


def audit_module():
    spec = importlib.util.spec_from_file_location("post418_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archived_audit_reproduces_all_observations():
    assert audit_module().analyze(EVIDENCE / "raw") == json.loads(
        (EVIDENCE / "summary.json").read_text()
    )


def test_optimized_audit_refuses_to_publish(tmp_path):
    output = tmp_path / "summary.json"
    result = subprocess.run(
        [sys.executable, "-O", str(SCRIPT), str(EVIDENCE / "raw"), str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "requires assertions" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "field", ["iteration_branch", "gpu4pyscf_sample_count", "speedup"]
)
def test_audit_rejects_corrupt_matched_summary(monkeypatch, field):
    """Recorded summaries cannot authorize an invented branch or stronger ratio."""
    audit = audit_module()
    original = audit.read

    def tampered_read(path):
        data = original(path)
        if path.name == "water-32mer-4s4-def2-svp-spherical-b1-forces.json":
            matched = data["timing_summary"]["iteration_matched"]
            matched[field] = [99] if field == "iteration_branch" else matched[field] + 1
        return data

    # Change only the interpreted derived field, retaining the same measured
    # arrays and their hash checks, to isolate the claim-validation boundary.
    monkeypatch.setattr(audit, "read", tampered_read)
    with pytest.raises(AssertionError, match="matched subset differs"):
        audit.analyze(EVIDENCE / "raw")
