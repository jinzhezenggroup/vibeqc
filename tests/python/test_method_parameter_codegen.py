"""Qualification for the audited method-parameter single source of truth."""

from __future__ import annotations

from vibeqc_compiler.method._generated_parameters import (
    D3_BJ_PARAMETER_SETS,
    D4_PARAMETER_SETS,
    GCP_PARAMETER_SETS,
    PARAMETER_SOURCE_SHA256,
)

from tools.generate_method_parameters import (
    DEFAULT_SOURCE,
    load_source,
    render_cpp,
    render_python,
)


def test_committed_python_parameters_are_fresh() -> None:
    payload, source_sha256 = load_source(DEFAULT_SOURCE)
    generated = DEFAULT_SOURCE.with_name("_generated_parameters.py")
    assert PARAMETER_SOURCE_SHA256 == source_sha256
    assert generated.read_text() == render_python(payload, source_sha256)


def test_cpp_codegen_uses_same_source_identity_and_values() -> None:
    payload, source_sha256 = load_source(DEFAULT_SOURCE)
    header = render_cpp(payload, source_sha256)
    assert source_sha256 in header
    assert "pbeD3BJ()" in header
    assert "pbe0D3BJ()" in header
    assert "r2scan3cD4()" in header
    assert "r2scan3cD4ChargeCnCutoff()" in header
    assert "r2scan3cGcp()" in header
    assert "r2scan3cGcpSupportsAtomicNumber" in header
    assert "0.7875" in header
    assert "1.2177" in header
    assert "5.65" in header


def test_generated_python_catalog_has_expected_profiles() -> None:
    assert set(D3_BJ_PARAMETER_SETS) == {"PBE-D3(BJ)", "PBE0-D3(BJ)"}
    assert set(D4_PARAMETER_SETS) == {"r2SCAN-3c"}
    assert set(GCP_PARAMETER_SETS) == {"r2SCAN-3c"}
    assert D4_PARAMETER_SETS["r2SCAN-3c"]["charge_cn_cutoff"] == 25.0
    assert GCP_PARAMETER_SETS["r2SCAN-3c"]["supported_atomic_numbers"] == tuple(
        range(1, 19)
    )
