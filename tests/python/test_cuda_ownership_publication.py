"""Verify lossless endpoint retention and reject corrupted retirement evidence."""

import ast
import gzip
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from tools.publish_cuda_ownership import compact_comparison, validate_resources, write
from tools.vibeqc_validation.publication import validate_publication

BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/cuda-ownership/one-electron"
)


def restore_workers(directory, *, one_case=False, bundle=BUNDLE):
    """Reconstruct original workers solely from permanent retained records."""
    compact = json.loads((bundle / "samples.json").read_text())
    records = compact["records"]
    for entry in compact["runs"]:
        run = dict(records[entry["provenance"]])
        run["endpoints"] = []
        for endpoint in entry["endpoints"][:1] if one_case else entry["endpoints"]:
            expanded = dict(records[endpoint["input"]])
            for key in ("resource_plan", "observed_resources", "density_fitting"):
                expanded[key] = records[endpoint[key]]
            expanded.update(seconds=endpoint["seconds"], results=endpoint["results"])
            if "energy_only" in endpoint:
                expanded["energy_only"] = endpoint["energy_only"]
            run["endpoints"].append(expanded)
        write(directory / f"{entry['selection']}-{entry['sample']}.json", run)
    write(
        directory / "comparison.json",
        {
            "samples": 5,
            "passed": True,
            "endpoint_ceiling": 1.02,
            **{
                key: records[compact["runs"][0]["provenance"]][key]
                for key in ("process_scope", "benchmark_driver_sha256")
                if key in records[compact["runs"][0]["provenance"]]
            },
        },
    )
    return compact


@pytest.mark.parametrize(
    "bundle,cases,error_blocks",
    [(BUNDLE, 20, 8), (BUNDLE.parent / "df", 18, 9)],
)
def test_retained_workers_round_trip_and_recompute_all_gates(
    tmp_path, bundle, cases, error_blocks
):
    original = restore_workers(tmp_path, bundle=bundle)
    compact, _, errors, _, rows = compact_comparison(tmp_path)
    assert compact == original
    assert len(rows) == cases
    assert len(errors) == cases * 10 * error_blocks
    summary = json.loads((bundle / "summary.json").read_text())
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


@pytest.mark.parametrize("bundle", [BUNDLE, BUNDLE.parent / "df"])
def test_published_checksums_and_decision(bundle):
    manifest = json.loads((bundle / "publication.json").read_text())
    files = {e["path"]: (bundle / e["path"]).read_bytes() for e in manifest["files"]}
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
@pytest.mark.parametrize(
    "bundle,resource_name,domain",
    [
        (BUNDLE, "candidate-pair", "one-electron"),
        (BUNDLE.parent / "df", "candidate-rhf", "df"),
    ],
)
def test_resources_must_match_the_measured_workers(
    tmp_path, field, bundle, resource_name, domain
):
    compact = restore_workers(tmp_path, one_case=True, bundle=bundle)
    workers = {
        row["selection"]: compact["records"][row["provenance"]]
        for row in compact["runs"]
    }
    resources = json.loads((bundle / "resources.json").read_text())
    validate_resources(
        resources, workers["baseline"], workers["candidate"], domain=domain
    )
    resources[resource_name]["provenance"][field] = "wrong-source-or-build"
    with pytest.raises(ValueError, match="resource/worker provenance mismatch"):
        validate_resources(
            resources, workers["baseline"], workers["candidate"], domain=domain
        )


def test_final_df_validation_is_bound_to_the_endpoint_library():
    """Recheck retained integration/oracle evidence independently of its archiver."""
    bundle = BUNDLE.parent / "df"
    manifest = json.loads((bundle / "validation-files.json").read_text())
    for row in manifest["files"]:
        data = (bundle / row["path"]).read_bytes()
        assert len(data) == row["bytes"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
    validation = json.loads((bundle / "validation.json").read_text())
    compact = json.loads((bundle / "samples.json").read_text())
    for run in compact["runs"]:
        if run["selection"] == "candidate":
            worker = compact["records"][run["provenance"]]
            assert validation["endpoint_revision"] == worker["revision"]
            for key in ("native_source_identity", "library_sha256"):
                assert validation["source"][key] == worker[key]
    assert (
        validation["validation_source"]["revision"] == validation["source"]["revision"]
    )
    assert len(validation["gpu_suite_drivers"]) == 6
    assert sum(row["tests"] for row in validation["gpu_suite_drivers"]) == 143
    inventory = validation["gpu_case_inventory"]
    assert len(inventory["collected"]) == len(set(inventory["collected"])) == 143
    assert sorted(inventory["collected"]) == sorted(inventory["executed"])
    for stage in validation["gpu_stage_sources"].values():
        for key in ("revision", "native_source_identity", "library_sha256"):
            assert stage[key] == validation["source"][key]
    raw = json.loads((bundle / "raw-source.json").read_text())
    assert raw["library_hash"] == validation["source"]["library_sha256"]
    assert len(raw["runs"]) == 24
    assert all(row["passed"] and row["ranks_match"] for row in raw["runs"])
    assert all(
        error["passed"] for row in raw["runs"] for error in row["errors"].values()
    )
    original = validation["original_files"]
    for row in original.values():
        data = row["text"].encode()
        assert len(data) == row["bytes"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
    for name, skipped in (("cpu", 2), ("runtime", 0), ("cuda", 0)):
        suites = list(
            ElementTree.fromstring(original[name + ".xml"]["text"]).iter("testsuite")
        )
        assert suites and sum(int(s.attrib["tests"]) for s in suites) > 0
        assert sum(int(s.attrib["skipped"]) for s in suites) == skipped
        assert all(
            int(s.attrib["errors"]) == int(s.attrib["failures"]) == 0 for s in suites
        )
    for backend, count in (("cpu", 15), ("cuda", 18)):
        assert validation["native_tests_passed"][backend] == count
        assert (
            f"100% tests passed, 0 tests failed out of {count}"
            in original[backend + "-native.log"]["text"]
        )
    for name in ("memcheck", "synccheck"):
        assert "ERROR SUMMARY: 0 errors" in original[name + ".log"]["text"]
        assert validation["sanitizers"][name]["tests_passed"] == 3


def test_rejected_inventory_run_is_lossless_and_still_fails_raw_gates(tmp_path):
    """Retain the failed complete matrix even after the paired matrix passes."""
    bundle = BUNDLE.parent / "df"
    manifest = json.loads((bundle / "rejected-inventory-run.json").read_text())
    assert manifest["decision"] == "rejected"
    assert manifest["promotion_evidence"] is False
    archive = manifest["archive"]
    data = (bundle / archive["path"]).read_bytes()
    assert len(data) == archive["bytes"]
    assert hashlib.sha256(data).hexdigest() == archive["sha256"]
    raw = gzip.decompress(data)
    assert len(raw) == archive["uncompressed_bytes"]
    assert hashlib.sha256(raw).hexdigest() == archive["uncompressed_sha256"]
    original = json.loads(raw)["original_files"]
    expected = {"comparison.json"} | {
        f"{selection}-{sample}.json"
        for selection in ("baseline", "candidate")
        for sample in range(5)
    }
    assert set(original) == set(manifest["original_files"]) == expected
    for name, row in original.items():
        data = row["text"].encode()
        assert len(data) == row["bytes"] == manifest["original_files"][name]["bytes"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        assert row["sha256"] == manifest["original_files"][name]["sha256"]
        (tmp_path / name).write_bytes(data)
        if name != "comparison.json":
            worker = json.loads(data)
            assert len(worker["endpoints"]) == 18
            assert worker["slurm_job_id"] == "9229"
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert comparison["passed"] is False
    failed = [
        {"case": row["case"], "phase": phase, "median_ratio": ratio}
        for row in comparison["endpoints"]
        for phase, ratio in row["phase_ratios"].items()
        if ratio > comparison["endpoint_ceiling"]
    ]
    assert failed == manifest["failed_gates"] and len(failed) == 1
    # Even an incorrectly accepted summary cannot hide the actual regression.
    comparison["passed"] = True
    write(tmp_path / "comparison.json", comparison)
    with pytest.raises(ValueError, match="raw numerical or nonregression gate failed"):
        compact_comparison(tmp_path)


def test_final_df_archive_reconstructs_all_original_case_process_bytes(tmp_path):
    """Bind the lossless aggregate to all 180 immutable measurement processes."""
    bundle = BUNDLE.parent / "df"
    restore_workers(tmp_path, bundle=bundle)
    manifest = json.loads((bundle / "process-files.json").read_text())
    assert manifest["process_scope"] == "case" and len(manifest["files"]) == 180
    identities = set()
    for row in manifest["files"]:
        identity = row["case"], row["selection"], row["sample"]
        assert identity not in identities
        identities.add(identity)
        aggregate = json.loads(
            (tmp_path / f"{row['selection']}-{row['sample']}.json").read_text()
        )
        aggregate["endpoints"] = [
            endpoint
            for endpoint in aggregate["endpoints"]
            if endpoint["case"] == row["case"]
        ]
        assert len(aggregate["endpoints"]) == 1
        assert (
            aggregate["benchmark_driver_sha256"] == manifest["benchmark_driver_sha256"]
        )
        data = (json.dumps(aggregate, sort_keys=True, indent=2) + "\n").encode()
        assert len(data) == row["bytes"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"]


def test_checkpoint_diagnosis_retains_failed_trials_and_exact_assertions():
    """The deterministic test setup must preserve the original bitwise gates."""
    bundle = BUNDLE.parent / "df"
    manifest = json.loads(
        (bundle / "checkpoint-validation-diagnostics.json").read_text()
    )
    archive = manifest["archive"]
    data = (bundle / archive["path"]).read_bytes()
    assert len(data) == archive["bytes"]
    assert hashlib.sha256(data).hexdigest() == archive["sha256"]
    data = gzip.decompress(data)
    assert len(data) == archive["uncompressed_bytes"]
    assert hashlib.sha256(data).hexdigest() == archive["uncompressed_sha256"]
    originals = json.loads(data)["original_files"]
    assert set(originals) == set(manifest["original_files"])
    for name, row in originals.items():
        data = row["text"].encode()
        assert len(data) == row["bytes"] == manifest["original_files"][name]["bytes"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        assert row["sha256"] == manifest["original_files"][name]["sha256"]
    for mode, job in (("default", "9236"), ("serial", "9237")):
        rows = json.loads(originals[f"checkpoint-roundoff-{job}/results.json"]["text"])
        assert len(rows) == 40
        for selection in ("baseline", "candidate"):
            trials = [row for row in rows if row["selection"] == selection]
            summary = manifest["diagnostic_trials"][mode][selection]
            assert len(trials) == summary["trials"] == 20
            assert sum(bool(row["exit_code"]) for row in trials) == summary["failures"]
            if mode == "serial":
                assert summary["failures"] == 0
            else:
                assert summary["failures"] > 0

    def body(text):
        return next(
            node.body
            for node in ast.parse(text).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "test_no_checkpoint_cold_run_remains_available_after_clear"
        )

    old = body(originals["original-checkpoint-test.py"]["text"])
    validated = originals["validated-checkpoint-test.py"]
    current = body(validated["text"])
    validation = json.loads((bundle / "validation.json").read_text())
    driver = next(
        row
        for row in validation["gpu_suite_drivers"]
        if row["source"] == "tests/python/test_checkpoint.py"
    )
    assert validated["sha256"] == driver["source_sha256"]
    # Only the explicit CUDA schedule setup precedes the original test body;
    # inputs, exact energy/force assertions and checkpoint-clear checks survive.
    assert isinstance(current[0], ast.If)
    assert [ast.dump(node) for node in current[1:]] == [ast.dump(node) for node in old]


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
