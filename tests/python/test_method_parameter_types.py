"""Typed generated parameter boundaries reject malformed source and runtime values."""

import json
from pathlib import Path

import pytest
from vibeqc_compiler.method import _generated_parameters as generated

from tools.generate_method_parameters import DEFAULT_SOURCE, load_source


@pytest.mark.parametrize(
    "category,field,value",
    [
        ("gcp", "damping", "false"),
        ("gcp", "damping", 1),
        ("gcp", "basis", 7),
        ("gcp", "eta", True),
        ("d4", "profile", 2),
        ("d4", "reference_model", False),
        ("d3_bj", "unexpected", 1.0),
    ],
)
def test_source_parameter_schema_rejects_type_erasure(
    tmp_path: Path,
    category: str,
    field: str,
    value: object,
) -> None:
    data = json.loads(DEFAULT_SOURCE.read_text())
    record = next(r for r in data[category] if category != "d4" or r.get("python_spec"))
    record["parameters"][field] = value
    path = tmp_path / "bad-parameters.json"
    path.write_text(json.dumps(data))
    with pytest.raises((TypeError, ValueError)):
        load_source(path)


def test_parameter_accessors_copy_validate_and_preserve_read_only_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for table, getter, name in (
        (generated.D3_BJ_PARAMETER_SETS, generated.d3_parameters, "PBE-D3(BJ)"),
        (generated.D4_PARAMETER_SETS, generated.d4_parameters, "r2SCAN-3c"),
        (generated.GCP_PARAMETER_SETS, generated.gcp_parameters, "r2SCAN-3c"),
    ):
        assert getter(name) == dict(table[name])
        detached = getter(name)
        detached.clear()
        assert getter(name) == dict(table[name])
    broken = dict(generated.GCP_PARAMETER_SETS["r2SCAN-3c"])
    broken["damping"] = 1
    monkeypatch.setattr(generated, "GCP_PARAMETER_SETS", {"r2SCAN-3c": broken})
    with pytest.raises(TypeError, match="bool"):
        generated.gcp_parameters("r2SCAN-3c")
