"""Default output destinations pass the same admission as explicit CLI paths."""

import argparse
import ast
from pathlib import Path

import pytest
from test_benchmark_output_retention import active_runner_output_arguments

from benchmarks import _retention, build_ledger


def test_build_ledger_default_rejects_retained_symlink_before_work(
    tmp_path, monkeypatch
):
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    (tmp_path / ".artifacts").symlink_to(retained, target_is_directory=True)
    original = retained / "benchmarks/build_ledger.json"
    original.parent.mkdir()
    original.write_text("original scientific evidence")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["build_ledger.py"])
    calls = []
    monkeypatch.setattr(
        build_ledger, "build_ledger", lambda *a, **k: calls.append(a) or {}
    )
    with pytest.raises(SystemExit) as error:
        build_ledger.main()
    assert error.value.code == 2
    assert calls == []
    assert original.read_text() == "original scientific evidence"


def test_live_output_defaults_are_strings_for_argparse_conversion():
    for path, node, options in active_runner_output_arguments():
        default = next((kw.value for kw in node.keywords if kw.arg == "default"), None)
        if default is None:
            continue
        constant_string = isinstance(default, ast.Constant) and isinstance(
            default.value, str
        )
        explicit_string = (
            isinstance(default, ast.Call)
            and isinstance(default.func, ast.Name)
            and default.func.id == "str"
        )
        assert constant_string or explicit_string, (path, options, ast.unparse(default))


def test_default_scratch_output_keeps_path_contract(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=_retention.raw_output_path, default=".artifacts/run.json"
    )
    assert parser.parse_args([]).output == Path(".artifacts/run.json")
    assert list(tmp_path.iterdir()) == []
