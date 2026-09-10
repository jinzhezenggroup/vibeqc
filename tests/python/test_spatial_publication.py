"""Audit permanent spatial evidence without native execution or historical builds."""

import copy
import json
from pathlib import Path

import pytest
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.resources import ResourcePlan, plan_resources

from tools.publish_spatial_tasks import dense_comparison, summarize, validate_run, write
from tools.vibeqc_validation.publication import validate_publication

BUNDLE = Path(__file__).resolve().parents[2] / "benchmarks/results/spatial-tasks"


def worker(backend):
    """Load a fresh worker so a corruption cannot leak into another case."""
    return json.loads((BUNDLE / backend / "samples.json").read_text())


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_published_inventory_resource_plans_and_summary(backend):
    root = BUNDLE / backend
    run = validate_run(worker(backend))
    assert len(run["cases"]) == 24
    stored = json.loads((root / "summary.json").read_text())
    stored.pop("historical_dense_comparison", None)
    assert summarize(run) == stored
    manifest = json.loads((root / "publication.json").read_text())
    files = {e["path"]: (root / e["path"]).read_bytes() for e in manifest["files"]}
    validate_publication(manifest, files)
    evidence = json.loads(files["evidence.json"])
    assert evidence["revision"] == run["revision"]
    assert evidence["hashes"]["source"] == run["source_identity"]
    assert evidence["toolchain"]["native_library_sha256"] == run["library_sha256"]
    assert manifest["decision"]["scope"] == "numerical"
    assert evidence["performance"]["status"] == "not-run"
    assert evidence["stages"]["production"]["status"] == "not-run"
    files["samples.json"] += b" "
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        validate_publication(manifest, files)


@pytest.mark.parametrize(
    "damage, message",
    [
        ("dirty", "clean worker"),
        ("inventory", "inventory"),
        ("mode", "execution metadata"),
        ("sample", "five complete"),
        ("tolerance", "arithmetic gates"),
        ("error", "arithmetic gates"),
        ("shape", "arithmetic gates"),
        ("nan", "arithmetic gates"),
        ("missing_timing", "execution timing"),
        ("negative_timing", "execution timing"),
        ("budget", "budget mismatch"),
        ("aggregate", "resource plan checksum"),
        ("capacity", "native capacity mismatch"),
        ("observation", "observation exceeds plan"),
        ("negative_observation", "observation exceeds plan"),
    ],
)
def test_worker_corruption_is_rejected(damage, message):
    run = worker("cuda")
    row = run["cases"][2]
    sample = row["samples"][0]
    if damage == "dirty":
        run["dirty"] = True
    elif damage == "inventory":
        run["cases"].pop()
    elif damage == "mode":
        row["mode"] = "dense"
    elif damage == "sample":
        row["samples"].pop()
    elif damage == "tolerance":
        sample["errors"]["rho"]["atol"] *= 10
    elif damage == "error":
        sample["errors"]["rho"]["max_scaled_error"] = 1.01
    elif damage == "shape":
        sample["errors"]["rho"]["shape"] = [1]
    elif damage == "nan":
        sample["errors"]["rho"]["max_absolute_error"] = float("nan")
    elif damage == "missing_timing":
        sample.pop("device_consumer_scatter_seconds")
    elif damage == "negative_timing":
        sample["features_seconds"] = -1
    elif damage == "budget":
        row["budgets"]["device_bytes"] *= 2
    elif damage == "aggregate":
        row["resource_plan"]["peak_bytes"]["device"] = 0
    elif damage == "capacity":
        row["tile_plan"]["allocation_bytes"] += 256
        sample["native_metrics"]["owned_device_bytes"] += 256
    elif damage == "observation":
        sample["native_metrics"]["owned_device_bytes"] += 256
    elif damage == "negative_observation":
        sample["native_metrics"]["provider_retained_bytes"] = -1
    with pytest.raises(ValueError, match=message):
        validate_run(run)


def test_missing_execution_request_has_a_publication_diagnostic():
    run = worker("cuda")
    row = run["cases"][2]
    plan = ResourcePlan.from_dict(row["resource_plan"])
    remaining = tuple(r for r in plan.requests if r.name != "spatial_execution")
    row["resource_plan"] = plan_resources(remaining, plan.budget).to_dict()
    with pytest.raises(ValueError, match="spatial execution candidate"):
        validate_run(run)


def test_empty_execution_candidates_are_rejected_as_invalid_data():
    run = worker("cuda")
    plan = run["cases"][2]["resource_plan"]
    next(r for r in plan["requests"] if r["name"] == "spatial_execution")[
        "candidates"
    ] = []
    plan.pop("identity")
    plan["identity"] = canonical_hash(plan)
    with pytest.raises(ValueError, match="neither a provider"):
        validate_run(run)


def restore_dense(directory):
    """Reconstruct all historical process workers solely from the permanent bundle."""
    retained = json.loads((BUNDLE / "cuda/dense-comparison-samples.json").read_text())
    for side, runs in retained["runs"].items():
        for index, run in enumerate(runs):
            write(directory / f"234-optimized-dense-{side}-{index}.json", run)
    return retained


def test_historical_dense_statistics_are_reproducible(tmp_path):
    retained = restore_dense(tmp_path)
    samples, rows = dense_comparison(tmp_path)
    assert samples == retained
    summary = json.loads((BUNDLE / "cuda/summary.json").read_text())
    assert rows == summary["historical_dense_comparison"]


@pytest.mark.parametrize("field", ["revision", "library_sha256", "inputs"])
def test_dense_comparison_rejects_changed_source_or_problem(tmp_path, field):
    retained = restore_dense(tmp_path)
    run = copy.deepcopy(retained["runs"]["candidate"][1])
    if field == "inputs":
        run["cases"][0]["inputs"]["density"] = "0" * 64
    else:
        run[field] = "0" * len(run[field])
    write(tmp_path / "234-optimized-dense-candidate-1.json", run)
    with pytest.raises(ValueError, match="source/build changes|inputs differ"):
        dense_comparison(tmp_path)
