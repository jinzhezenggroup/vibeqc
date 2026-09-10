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


def synthetic_df_workers(directory):
    """Extend reconstructed fixtures solely to exercise the publication schema.

    These CPU-only test inputs are never published as measured DF evidence.
    """
    restore_workers(directory, one_case=True)
    for selection in ("baseline", "candidate"):
        for index in range(5):
            path = directory / f"{selection}-{index}.json"
            run = json.loads(path.read_text())
            run["domain"] = "df"
            endpoint = run["endpoints"][0]
            endpoint["energy_only"] = {
                "properties": ["energy"],
                "seconds": 0.1,
                "results": [
                    {k: v for k, v in r.items() if k != "forces"}
                    for r in endpoint["results"]["warm"]
                ],
            }
            write(path, run)


def test_df_publication_retains_and_gates_energy_only_calls(tmp_path):
    synthetic_df_workers(tmp_path)
    compact, _, errors, timings, rows = compact_comparison(tmp_path)
    assert len(errors) == 10 * (4 * 2 + 1)
    assert rows[0]["phase_ratios"]["energy-only-singlepoints"] == 1
    assert len([t for t in timings if t["phase"] == "energy-only"]) == 10
    for run in compact["runs"]:
        original = json.loads(
            (tmp_path / f"{run['selection']}-{run['sample']}.json").read_text()
        )
        assert (
            run["endpoints"][0]["energy_only"]
            == original["endpoints"][0]["energy_only"]
        )


@pytest.mark.parametrize(
    "corruption,match",
    [
        ("slow", "numerical or nonregression"),
        ("energy", "numerical or nonregression"),
        ("negative", "shared timing validation failed"),
        ("missing", "requires every energy-only"),
        ("properties", "energy-only properties"),
        ("iterations", "energy-only residual/count"),
    ],
)
def test_df_summary_cannot_hide_energy_only_failure(tmp_path, corruption, match):
    synthetic_df_workers(tmp_path)
    for index in range(5):
        path = tmp_path / f"candidate-{index}.json"
        run = json.loads(path.read_text())
        endpoint = run["endpoints"][0]
        value = endpoint["energy_only"]
        if corruption == "slow":
            value["seconds"] *= 1.03
        elif corruption == "energy":
            value["results"][0]["energy"] += 1e-4
        elif corruption == "negative":
            value["seconds"] = -1
        elif corruption == "missing":
            del endpoint["energy_only"]
        elif corruption == "properties":
            value["properties"].append("forces")
        elif corruption == "iterations":
            value["results"][0]["iterations"] = 0
        write(path, run)
    with pytest.raises(ValueError, match=match):
        compact_comparison(tmp_path)


@pytest.mark.parametrize("corruption", ["process_scope", "benchmark_driver_sha256"])
def test_paired_publication_retains_and_checks_measurement_contract(
    tmp_path, corruption
):
    synthetic_df_workers(tmp_path)
    for path in tmp_path.glob("*-*.json"):
        run = json.loads(path.read_text())
        run.update(process_scope="case", benchmark_driver_sha256="0" * 64)
        write(path, run)
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    comparison.update(process_scope="case", benchmark_driver_sha256="0" * 64)
    write(tmp_path / "comparison.json", comparison)
    compact, _, _, _, _ = compact_comparison(tmp_path)
    assert all(
        compact["records"][run["provenance"]]["process_scope"] == "case"
        for run in compact["runs"]
    )
    path = tmp_path / "candidate-4.json"
    run = json.loads(path.read_text())
    run[corruption] = "inventory" if corruption == "process_scope" else "1" * 64
    write(path, run)
    with pytest.raises(ValueError, match="process scope or benchmark driver"):
        compact_comparison(tmp_path)
