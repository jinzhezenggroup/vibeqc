"""Quoted TOML keys are method names, not literal quote characters."""

from pathlib import Path

import pytest
from vibeqc_compiler.method import _generated_parameters

from tools.sync_dispersion_parameters import _parse_variant, build_catalog


@pytest.mark.parametrize("version", ("1.0", "1.1"))
def test_pinned_skala_parameters_use_unquoted_lookup_names(version: str) -> None:
    name = f"SKALA-{version}-D3(BJ)"
    parameters = _generated_parameters.d3_parameters(name)
    record = next(r for r in build_catalog()["d3_bj"] if r["name"] == name)
    assert record["provenance"]["upstream_key"] == f"skala-{version}"
    assert parameters["a1"] == record["parameters"]["a1"]


@pytest.mark.parametrize("key", ('"method.1"', "'method.1'"))
def test_toml_quoted_method_key_is_decoded(tmp_path: Path, key: str) -> None:
    path = tmp_path / "parameters.toml"
    path.write_text(
        "[default.parameter]\nd3.bj = {s6=1, s8=1, a1=0.4, a2=4}\n"
        f"[parameter.{key}]\nd3.bj = {{s8=2}}\n"
    )
    actual = _parse_variant(path, "d3.bj")
    assert list(actual) == ["method.1"]
    assert actual["method.1"]["s8"] == 2
