"""Retained larger D/C evidence must keep paired workloads and bounded execution."""

import copy
import json
from hashlib import sha256
from pathlib import Path

import pytest

from tools.density_workload_matrix import validate_matrix_errors
from tools.summarize_density_candidates import summarize
from tools.vibeqc_validation.publication import validate_publication

ROOT = Path(__file__).resolve().parents[2] / "benchmarks/results/density-candidates"


@pytest.fixture(scope="module", params=["gpu", "native"])
def publication(request):
    directory = ROOT / request.param
    manifest = json.loads((directory / "publication.json").read_text())
    files = {r["path"]: (directory / r["path"]).read_bytes() for r in manifest["files"]}
    validate_publication(manifest, files)
    filename = next(r["path"] for r in manifest["files"] if r["role"] == "evidence")
    return request.param, json.loads(files[filename]), files


def test_retained_summary_reconstructs_and_has_complete_matrix(publication):
    kind, report, files = publication
    summary = summarize(report)
    assert summary == json.loads(files["summary.json"])
    assert report["dirty"] is False
    assert all(row["passed"] for row in summary["errors"].values())
    assert report["stages"]["production"]["status"] == "not-run"
    if kind == "native":
        assert len(report["timings"]) == 80
        assert len(summary["cases"]) == 8
        assert sha256(files["samples.csv"]).hexdigest() == report["samples_sha256"]
        assert report["device"]["cpu"]
        for row in report["timings"]:
            diagnostic = row["diagnostics"]
            assert diagnostic["fallbacks"] == int(
                row["selection"] == "candidate"
                and diagnostic["scenario"] == "common-warm-density"
            )
            assert (
                diagnostic["orbital_calls"] > 0
                if row["selection"] == "candidate"
                else diagnostic["orbital_calls"] == 0
            )
    else:
        validate_matrix_errors(report["block_errors"])
        assert len(report["endpoint_cases"]) == 36
        assert len(report["batch_cases"]) == 12
        assert len(report["timings"]) == 480
        assert summary["block_gate_count"] == 392
        assert {r["nao"] for r in report["endpoint_cases"]} == {24, 43, 93, 96, 192}
        for case in report["endpoint_cases"] + report["batch_cases"]:
            plan = case["resource_plan"]
            assert plan["status"] == "feasible"
            for space in ("host", "device"):
                assert plan["peak_bytes"][space] <= plan["limits"][space]
        assert {
            r["resource_plan"]["limits"]["device"] for r in report["endpoint_cases"]
        } == {128 << 20, 256 << 20}
        for row in report["timings"]:
            diagnostic = row["diagnostics"]
            for item in diagnostic.get("items", [diagnostic]):
                route = (
                    "density_matrix" if row["selection"] == "baseline" else "orbitals"
                )
                assert item["executed_route"] == item["requested_route"] == route
                assert item["statistics"]["source"]["source_kind"] == route


@pytest.mark.parametrize("fault", ["missing", "duplicate", "identity", "nonpositive"])
def test_summary_rejects_incomparable_or_incomplete_pairs(publication, fault):
    _, report, _ = publication
    report = copy.deepcopy(report)
    if fault == "missing":
        report["timings"].pop()
    elif fault == "duplicate":
        report["timings"].append(report["timings"][0])
    elif fault == "identity":
        report["timings"][0]["inputs_hash"] = "a" * 64
    else:
        report["timings"][0]["seconds"] = 0
    with pytest.raises(ValueError):
        summarize(report)


@pytest.mark.parametrize("rename", [False, True])
def test_matrix_gate_inventory_rejects_missing_or_reused_keys(rename):
    report = json.loads((ROOT / "gpu/evidence.json").read_text())
    key = next(iter(report["block_errors"]))
    record = report["block_errors"].pop(key)
    if rename:
        report["block_errors"][key + "_wrong"] = record
        assert len(report["block_errors"]) == 392  # A count-only gate would pass.
    with pytest.raises(AssertionError, match="workload/error inventory"):
        validate_matrix_errors(report["block_errors"])
