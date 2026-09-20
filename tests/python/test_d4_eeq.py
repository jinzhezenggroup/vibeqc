"""D4 EEQ MethodIR and pinned-data identity tests for #493."""

import json
import typing
from dataclasses import replace
from pathlib import Path

import pytest
from vibeqc_compiler.method import (
    METHOD_CATALOG,
    D4Spec,
    DispersionCorrectionPrimitive,
    r2scan3c_d4_eeq,
    resolve_method,
)


def test_r2scan3c_d4_manifest_is_exact_and_roundtrips() -> None:
    spec = r2scan3c_d4_eeq()
    assert D4Spec(**spec.to_payload()) == spec
    assert (spec.s6, spec.s8, spec.s9, spec.a1, spec.a2) == (
        1.0,
        0.0,
        2.0,
        0.42,
        5.65,
    )
    assert (spec.ga, spec.gc, spec.profile) == (2.0, 1.0, "r2scan3c")
    assert (spec.cn_cutoff, spec.pair_cutoff, spec.atm_cutoff) == (30.0, 60.0, 40.0)
    assert spec.charge_cn_cutoff == 25.0


def test_r2scan3c_manifest_hashes_generated_assets() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "src/dft/dispersion/d4_eeq_manifest.json").read_text()
    )
    spec = r2scan3c_d4_eeq()
    assert spec.table_sha256 == manifest["bundle_sha256"]
    assert (
        spec.charge_parameter_sha256 == manifest["sources"]["multicharge"][0]["sha256"]
    )


def test_d4_composes_without_method_specific_scientific_node() -> None:
    correction = r2scan3c_d4_eeq()
    graph = resolve_method(replace(METHOD_CATALOG["PBE"], dispersion=correction))
    assert graph.primitives[-1] == DispersionCorrectionPrimitive(correction)
    assert graph.requirements["operators"] == ("semilocal-xc", "geometry-d4-bj-eeq")
    changed = resolve_method(
        replace(METHOD_CATALOG["PBE"], dispersion=replace(correction, s8=0.25))
    )
    assert changed.identity != graph.identity


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"profile": "standard"}, "profile and zeta"),
        ({"ga": 3.0}, "profile and zeta"),
        ({"charge_model": "gfn2"}, "only the pinned"),
        ({"table_sha256": "0"}, "SHA-256"),
    ],
)
def test_d4_manifest_rejects_semantic_mismatch(
    changes: typing.Any, match: typing.Any
) -> None:
    with pytest.raises(ValueError, match=match):
        replace(r2scan3c_d4_eeq(), **changes)


def test_d4_and_nonlocal_correlation_survive_shared_method_composition() -> None:
    from vibeqc_compiler.method import VV10, original_nonlocal_correlation

    correction = r2scan3c_d4_eeq()
    nonlocal_correlation = original_nonlocal_correlation(VV10)
    combined = replace(
        METHOD_CATALOG["PBE"],
        dispersion=correction,
        nonlocal_correlation=nonlocal_correlation,
    )
    graph = resolve_method(combined)
    assert graph.requirements["operators"] == (
        "semilocal-xc",
        "nonlocal-correlation",
        "geometry-d4-bj-eeq",
    )
    assert graph.identity != resolve_method(replace(combined, dispersion=None)).identity
    assert (
        graph.identity
        != resolve_method(replace(combined, nonlocal_correlation=None)).identity
    )
