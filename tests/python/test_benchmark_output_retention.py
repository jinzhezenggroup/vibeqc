"""Execution and deliberate evidence publication have different output paths."""

from __future__ import annotations

import argparse
import ast
import typing
from pathlib import Path

import pytest

from benchmarks import _support

ROOT = Path(__file__).resolve().parents[2]


def active_runner_output_arguments() -> typing.Any:
    """Yield every live benchmark CLI output path, excluding frozen evidence scripts."""

    benchmark_root = ROOT / "benchmarks"
    for path in sorted(benchmark_root.rglob("*.py")):
        relative = path.relative_to(benchmark_root)
        if "results" in relative.parts or path.name == "_support.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Attribute)
                or node.func.attr != "add_argument"
            ):
                continue
            options = [
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ]
            if not any(
                option == "--output"
                or option.endswith(
                    (
                        "-output",
                        "-output-dir",
                        "-output-directory",
                        "-output-file",
                        "-output-path",
                    )
                )
                for option in options
            ):
                continue
            yield path, node, options


@pytest.mark.parametrize("alias", [False, True])
def test_raw_writer_rejects_retained_tree_and_symlink_alias(
    tmp_path: typing.Any, monkeypatch: typing.Any, alias: typing.Any
) -> None:
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
    tmp_path: typing.Any, monkeypatch: typing.Any
) -> None:
    monkeypatch.setattr(_support, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="raw benchmark output"):
        _support.write_result(".artifacts/../benchmarks/results/new.json", {})
    assert not (tmp_path / "benchmarks").exists()


def test_scratch_paths_and_neighbour_names_remain_supported(
    tmp_path: typing.Any, monkeypatch: typing.Any
) -> None:
    monkeypatch.setattr(_support, "_REPOSITORY_ROOT", tmp_path)
    for name in (
        ".artifacts/benchmarks/run.json",
        "benchmarks/results-copy/a.json",
        "scratch/run.json",
    ):
        path = tmp_path / name
        assert _support.write_result(path, {"samples": [1, 2]}) == path
        assert path.is_file()


def test_argparse_reports_bad_output_without_starting_work(
    tmp_path: typing.Any, monkeypatch: typing.Any
) -> None:
    monkeypatch.setattr(_support, "_REPOSITORY_ROOT", tmp_path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=_support.raw_output_path)
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--output", str(tmp_path / "benchmarks/results/x.json")])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_all_active_runner_outputs_are_guarded_during_argument_parsing() -> None:
    seen = []
    for path, node, options in active_runner_output_arguments():
        seen.append((path.relative_to(ROOT), tuple(options)))
        keywords = {kw.arg: kw.value for kw in node.keywords}
        assert isinstance(keywords.get("type"), ast.Name), (path, options)
        assert keywords["type"].id == "raw_output_path", (path, options)
        default = keywords.get("default")
        if isinstance(default, ast.Constant) and isinstance(default.value, str):
            assert default.value.startswith(".artifacts/"), (
                path,
                options,
                default.value,
            )
    assert seen
    assert (Path("benchmarks/h2_latency.py"), ("--output",)) in seen
    assert (
        Path("benchmarks/compare_gpu4pyscf_batch.py"),
        ("--progress-output",),
    ) in seen
    assert (
        Path("benchmarks/experiments/issue409-packed-values/run_endpoints.py"),
        ("--output",),
    ) in seen
