"""The retained CPU oracle adapter must preserve evidence admission boundaries."""

import gzip
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ADAPTER = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/issue949-matched-768-20260923/reuse_cpu_oracle.py"
)


def load_adapter():
    spec = importlib.util.spec_from_file_location("issue949_cpu_oracle", ADAPTER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("version", ["2.13.0", None])
def test_retained_oracle_requires_exact_pyscf_version(version, monkeypatch, tmp_path):
    adapter = load_adapter()
    with gzip.open(adapter.RECEIPT, "rt") as stream:
        receipt = json.load(stream)
    if version is None:
        receipt["cpu_reference"].pop("pyscf_version")
    else:
        receipt["cpu_reference"]["pyscf_version"] = version
    changed = tmp_path / "changed.json.gz"
    with gzip.open(changed, "wt") as stream:
        json.dump(receipt, stream)
    monkeypatch.setattr(adapter, "RECEIPT", changed)
    case = adapter.endpoint.benchmark_cases()[adapter.endpoint.CASES[768]]
    with pytest.raises(RuntimeError, match="oracle_version"):
        adapter.retained_cpu_reference(case, case.pyscf_basis, case.pyscf_basis)


def test_existing_output_remains_byte_identical(monkeypatch, tmp_path):
    adapter = load_adapter()
    output = tmp_path / "previous.json"
    previous = b'{"old":true, "format":"unchanged"}\n'
    output.write_bytes(previous)
    monkeypatch.setattr(sys, "argv", ["adapter", "--output", str(output)])

    def should_not_run():
        raise AssertionError("endpoint should not run with a pre-existing output")

    monkeypatch.setattr(adapter.endpoint, "main", should_not_run)
    with pytest.raises(RuntimeError, match="refusing to overwrite evidence"):
        adapter.main()
    assert output.read_bytes() == previous
