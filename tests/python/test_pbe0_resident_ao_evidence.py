"""Independent evidence gates must reject semantic corruption as well as hashes."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=["frozen", "current"])
def verifier(request: pytest.FixtureRequest) -> ModuleType:
    path = ROOT / "benchmarks/results/pbe0-resident-active-ao-20261003/verify.py"
    spec = importlib.util.spec_from_file_location("resident_ao_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_bundle(request.param)
    return module


def test_all_complete_endpoints_work_and_zero_budget_control(
    verifier: ModuleType,
) -> None:
    summary = verifier.verify_all()
    assert summary == verifier.load(Path("receipts/ao-summary.json"))
    assert summary["native_endpoints_checked"] == 156
    assert summary["cases"][-1]["atoms"] == 96
    assert summary["cases"][-1]["force_contraction_G_M2_fraction"] < 0.06


def test_current_device_qualification_does_not_relabel_frozen_timing() -> None:
    directory = ROOT / "benchmarks/results/pbe0-resident-active-ao-20261003"
    current = json.loads((directory / "current-qualification.json").read_text())
    frozen = json.loads((directory / "storage.json").read_text())
    assert current["source_identity"] != frozen["source_identity"]
    assert current["library_sha256"] != frozen["library_sha256"]
    stored = (directory / current["log"]).read_bytes()
    assert len(stored) == current["stored_bytes"]
    assert hashlib.sha256(stored).hexdigest() == current["stored_sha256"]
    raw = gzip.decompress(stored)
    assert hashlib.sha256(raw).hexdigest() == current["log_sha256"]
    log = raw.decode()
    assert f"job={current['job']} host=node1 devices=0" in log
    assert "SOURCE IDENTITY PASS " + current["source_identity"] in log
    assert "89 passed" in log and current["host_gpu_caller_tests_passed"] == 89
    assert "34 passed" in log and current["memcheck_tests_passed"] == 34
    assert "ERROR SUMMARY: 0 errors" in log and current["memcheck_errors"] == 0
    assert "4 passed" in log and current["claim_tests_passed"] == 4


@pytest.mark.parametrize(
    "corruption",
    ["force", "repeat", "pair_work", "discovery", "budget", "source"],
)
def test_semantic_gates_do_not_trust_retained_pass_flags(
    verifier: ModuleType, corruption: str
) -> None:
    directory = Path("endpoints/3")
    record_path = directory / "local.json"
    campaign_path = directory / "local.campaign.json"
    record = verifier.load(record_path)
    campaign = verifier.load(campaign_path)
    call = campaign["force_calls"][0]
    if corruption == "force":
        record["records"][0]["forces"][0][0] += 1e-4
    elif corruption == "repeat":
        record["records"].pop()
    elif corruption == "pair_work":
        call["grid_pair_visits"] -= 1
    elif corruption == "discovery":
        call["resident_ao_selection"]["work"]["discoveries"] = 0
    elif corruption == "budget":
        work = call["resident_ao_selection"]["work"]
        work["numeric_peak_bound_bytes"] = work["budget_bytes"] + 1
    else:
        campaign["source_identity"] = "0" * 64
    verifier._MEMBERS[record_path.as_posix()] = json.dumps(record)
    verifier._MEMBERS[campaign_path.as_posix()] = json.dumps(campaign)
    reference_path = directory / "reference.json"
    with pytest.raises(AssertionError):
        verifier.verify_variant(
            directory,
            3,
            "local",
            verifier.load(reference_path),
            verifier.digest(reference_path),
        )
