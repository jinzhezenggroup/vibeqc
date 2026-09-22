"""Pinned upstream dispersion catalogs regenerate the committed method source."""

from __future__ import annotations

import json

from tools.sync_dispersion_parameters import (
    DEFAULT_OUTPUT,
    SOURCE_MANIFEST,
    build_catalog,
    render_catalog,
)


def test_committed_dispersion_catalog_is_fresh() -> None:
    assert DEFAULT_OUTPUT.read_text() == render_catalog()


def test_pinned_upstream_catalog_counts_and_representative_values() -> None:
    payload = build_catalog()
    assert len(payload["d3_bj"]) == 157
    assert sum(bool(record.get("python_spec")) for record in payload["d4"]) == 118

    d3 = {record["name"]: record for record in payload["d3_bj"]}
    d4 = {
        record["name"]: record for record in payload["d4"] if record.get("python_spec")
    }

    assert d3["B3LYP-D3(BJ)"]["parameters"] == {
        "s6": 1.0,
        "s8": 1.9889,
        "a1": 0.3981,
        "a2": 4.4211,
        "s9": 0.0,
    }
    assert d3["R2SCAN-D3(BJ)"]["parameters"]["a2"] == 5.73083694
    assert d3["WB97M-D3(BJ)"]["parameters"]["s8"] == 0.3908
    assert all(record["parameters"]["s9"] == 0.0 for record in payload["d3_bj"])
    assert all(
        record["provenance"]["projection"] == "two-body-s9=0"
        for record in payload["d3_bj"]
    )

    assert d4["PBE-D4(BJ-EEQ-ATM)"]["parameters"]["s8"] == 0.95948085
    assert d4["B3LYP-D4(BJ-EEQ-ATM)"]["parameters"]["a2"] == 4.53807137
    assert d4["R2SCAN-D4(BJ-EEQ-ATM)"]["parameters"]["s8"] == 0.60187490
    assert d4["r2SCAN-3c"]["parameters"]["s9"] == 2.0
    assert d4["r2SCAN-3c"]["parameters"]["ga"] == 2.0
    assert d4["r2SCAN-3c"]["parameters"]["gc"] == 1.0


def test_catalog_provenance_matches_pinned_source_manifest() -> None:
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    payload = build_catalog()
    d3_source = manifest["simple_dftd3"]
    d4_source = manifest["dftd4"]

    for record in payload["d3_bj"]:
        provenance = record["provenance"]
        assert provenance["revision"] == d3_source["revision"]
        assert provenance["sha256"] == d3_source["sha256"]
        assert provenance["variant"] == "d3.bj"

    for record in payload["d4"]:
        if not record.get("python_spec"):
            continue
        provenance = record["provenance"]
        assert provenance["revision"] == d4_source["revision"]
        assert provenance["sha256"] == d4_source["sha256"]
        assert provenance["variant"] == "d4.bj-eeq-atm"
