"""Retained native endpoint evidence must reconstruct its quantitative decision."""

import copy
import json
from pathlib import Path

import pytest

from tools.publish_xc_contractions import summarize, validate_run
from tools.vibeqc_validation.publication import validate_publication

ROOT = Path(__file__).resolve().parents[2] / "benchmarks/results/xc-contractions"


@pytest.fixture(scope="module")
def run():
    return json.loads((ROOT / "samples.json").read_text())


def test_retained_xc_publication_is_complete_and_reconstructs(run):
    assert summarize(run) == json.loads((ROOT / "summary.json").read_text())
    manifest = json.loads((ROOT / "publication.json").read_text())
    validate_publication(
        manifest,
        {
            entry["path"]: (ROOT / entry["path"]).read_bytes()
            for entry in manifest["files"]
        },
    )
    assert len(run["cases"]) == 128
    assert sum(len(r["samples"]) for r in run["cases"]) == 640


@pytest.mark.parametrize(
    "fault",
    [
        "dirty",
        "case",
        "program",
        "weight_source",
        "raw_derivative",
        "counter",
        "gate",
        "timing",
        "plan",
        "local_mask",
    ],
)
def test_xc_publication_rejects_corrupt_scientific_evidence(run, fault):
    broken = copy.deepcopy(run)
    row = broken["cases"][0]
    if fault == "dirty":
        broken["dirty"] = True
    elif fault == "case":
        broken["cases"].pop()
    elif fault == "program":
        broken["programs"].pop("PBE/response")
    elif fault in ("weight_source", "raw_derivative"):
        values = broken["independent_derivatives"]["h2/PBE"]["projections"]
        if fault == "weight_source":
            values[:] = [p for p in values if p["axis"] != "weights"]
        else:
            values[0]["plus"] += 0.01
    elif fault == "counter":
        row["samples"][0]["statistics"]["total_matrix_products"] += 1
    elif fault == "gate":
        row["samples"][0]["errors"]["external_energy"]["atol"] = 1e-3
    elif fault == "timing":
        row["samples"][0]["execute_seconds"] = -1
    elif fault == "plan":
        row["resource_plan"] = broken["cases"][1]["resource_plan"]
    elif fault == "local_mask":
        for value in broken["cases"]:
            if value["fixture"] == "separated_f" and value["local"]:
                value["active_ao_sizes"] = [value["nao"]]
    with pytest.raises(ValueError):
        validate_run(broken)
