"""Execution and deliberate evidence publication have different output paths."""

import argparse
import ast
from pathlib import Path

import pytest

from benchmarks import _support

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = (
    "h2_latency.py",
    "batch_throughput.py",
    "compare_gpu4pyscf.py",
    "compare_gpu4pyscf_batch.py",
    "compare_df_exchange.py",
    "inactive_eigensolver_profile.py",
)


@pytest.mark.parametrize("alias", [False, True])
def test_raw_writer_rejects_retained_tree_and_symlink_alias(
    tmp_path, monkeypatch, alias
):
    monkeypatch.setattr(_support, "_REPOSITORY_ROOT", tmp_path)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    old = retained / "existing.json"
    old.write_text("original evidence")
    destination = retained
    if alias:
        destination = tmp_path / "alias"
        destination.symlink_to(retained, target_is_directory=True)
    with pytest.raises(ValueError, match="raw benchmark output"):
        _support.write_result(destination / "existing.json", {"overwrite": True})
    assert old.read_text() == "original evidence"
    with pytest.raises(ValueError, match="raw benchmark output"):
        _support.write_result(destination / "new/result.json", {"value": 1})
    assert not (retained / "new").exists()


def test_guard_rejects_relative_traversal_before_creating_anything(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(_support, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="raw benchmark output"):
        _support.write_result(".artifacts/../benchmarks/results/new.json", {})
    assert not (tmp_path / "benchmarks").exists()


def test_scratch_paths_and_neighbour_names_remain_supported(tmp_path, monkeypatch):
    monkeypatch.setattr(_support, "_REPOSITORY_ROOT", tmp_path)
    for name in (
        ".artifacts/benchmarks/run.json",
        "benchmarks/results-copy/a.json",
        "scratch/run.json",
    ):
        path = tmp_path / name
        assert _support.write_result(path, {"samples": [1, 2]}) == path
        assert path.is_file()


def test_argparse_reports_bad_output_without_starting_work(tmp_path, monkeypatch):
    monkeypatch.setattr(_support, "_REPOSITORY_ROOT", tmp_path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=_support.raw_output_path)
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--output", str(tmp_path / "benchmarks/results/x.json")])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", RUNNERS)
def test_shared_runner_outputs_are_guarded_during_argument_parsing(name):
    tree = ast.parse((ROOT / "benchmarks" / name).read_text())
    found = set()
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Attribute)
            or node.func.attr != "add_argument"
        ):
            continue
        options = [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
        if "--output" not in options and "--progress-output" not in options:
            continue
        found.update(options)
        keywords = {kw.arg: kw.value for kw in node.keywords}
        assert isinstance(keywords.get("type"), ast.Name)
        assert keywords["type"].id == "raw_output_path"
        if "default" in keywords:
            assert keywords["default"].value.startswith(".artifacts/")
    assert "--output" in found
