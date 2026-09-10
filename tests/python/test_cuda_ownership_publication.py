"""Verify lossless endpoint retention and reject corrupted retirement evidence."""

import json
from pathlib import Path

import pytest

from tools.publish_cuda_ownership import compact_comparison, validate_resources, write
from tools.vibeqc_validation.publication import validate_publication

BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/cuda-ownership/one-electron"
)


def restore_workers(directory, *, one_case=False):
    """Reconstruct original workers solely from permanent retained records."""
    compact = json.loads((BUNDLE / "samples.json").read_text())
    records = compact["records"]
    for entry in compact["runs"]:
        run = dict(records[entry["provenance"]])
        run["endpoints"] = []
        for endpoint in entry["endpoints"][:1] if one_case else entry["endpoints"]:
            expanded = dict(records[endpoint["input"]])
            for key in ("resource_plan", "observed_resources", "density_fitting"):
                expanded[key] = records[endpoint[key]]
            expanded.update(seconds=endpoint["seconds"], results=endpoint["results"])
            run["endpoints"].append(expanded)
        write(directory / f"{entry['selection']}-{entry['sample']}.json", run)
    write(
        directory / "comparison.json",
        {"samples": 5, "passed": True, "endpoint_ceiling": 1.02},
    )
    return compact


def test_retained_workers_round_trip_and_recompute_all_gates(tmp_path):
    original = restore_workers(tmp_path)
    compact, _, errors, _, rows = compact_comparison(tmp_path)
    assert compact == original
    assert len(rows) == 20
    assert len(errors) == 20 * 10 * 4 * 2
    summary = json.loads((BUNDLE / "summary.json").read_text())
    assert rows == summary["endpoints"]
    assert all(e["passed"] for e in errors.values())


@pytest.mark.parametrize(
    "corruption, match",
    [
        ("energy", "numerical or nonregression"),
        ("forces", "numerical or nonregression"),
        ("negative_prepare", "timing interval"),
        ("slow_warm", "numerical or nonregression"),
        ("source", "changing measured source"),
        ("dirty", "dirty or changing"),
        ("inputs", "mathematical inputs differ"),
        ("selection", "provenance"),
        ("fast_compile", "provenance"),
    ],
)
def test_accepted_summary_cannot_hide_corrupted_worker(tmp_path, corruption, match):
    restore_workers(tmp_path, one_case=True)
    for index in range(5):
        path = tmp_path / f"candidate-{index}.json"
        run = json.loads(path.read_text())
        endpoint = run["endpoints"][0]
        if corruption == "energy":
            endpoint["results"]["warm"][0]["energy"] += 1e-4
        elif corruption == "forces":
            endpoint["results"]["moved"][0]["forces"][0][0] += 1e-4
        elif corruption == "negative_prepare":
            endpoint["seconds"]["prepare"] = -1e-12
        elif corruption == "slow_warm":
            endpoint["seconds"]["warm"] *= 2
        elif corruption == "source" and index == 1:
            run["library_sha256"] = "0" * 64
        elif corruption == "dirty":
            run["dirty"] = True
        elif corruption == "inputs":
            endpoint["df_budget_bytes"] = 1
        elif corruption == "selection":
            run["selection"] = "reference"
        elif corruption == "fast_compile":
            run["build_settings"] = ["CMAKE_BUILD_TYPE:STRING=Release"]
        write(path, run)
    with pytest.raises(ValueError, match=match):
        compact_comparison(tmp_path)


def test_published_checksums_and_decision():
    manifest = json.loads((BUNDLE / "publication.json").read_text())
    files = {e["path"]: (BUNDLE / e["path"]).read_bytes() for e in manifest["files"]}
    validate_publication(manifest, files)
    assert manifest["decision"]["scope"] == "numerical"
    evidence = json.loads(files["evidence.json"])
    assert evidence["performance"]["status"] == "not-run"
    assert evidence["stages"]["production"]["status"] == "not-run"
    files["samples.json"] += b" "
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        validate_publication(manifest, files)


@pytest.mark.parametrize(
    "field", ["revision", "native_source_identity", "library_sha256", "build_settings"]
)
def test_resources_must_match_the_measured_workers(tmp_path, field):
    compact = restore_workers(tmp_path, one_case=True)
    workers = {
        row["selection"]: compact["records"][row["provenance"]]
        for row in compact["runs"]
    }
    resources = json.loads((BUNDLE / "resources.json").read_text())
    validate_resources(resources, workers["baseline"], workers["candidate"])
    resources["candidate-pair"]["provenance"][field] = "wrong-source-or-build"
    with pytest.raises(ValueError, match="resource/worker provenance mismatch"):
        validate_resources(resources, workers["baseline"], workers["candidate"])
