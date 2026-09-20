"""Crash visibility and incomplete-evidence contracts require no GPU context."""

import json
import os
import shutil
import subprocess
import typing
from pathlib import Path

import pytest

from benchmarks.df_progress_ledger import read_progress, summarize_progress


def test_live_journal_survives_process_exit_without_destructors(
    tmp_path: typing.Any,
) -> None:
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "probe.cpp"
    source.write_text(
        r"""
#include "runtime/df_progress_trace.hpp"
#include <cstdlib>
int main() {
  using namespace vibeqc::runtime::df_progress;
  Scope endpoint("endpoint");
  { Scope setup("setup"); number("nbf", 768); }
  Scope raw("raw_materialization");
  label("provider", "generated");
  std::_Exit(0);  // Models timeout/SIGKILL: no scope destructor can flush BEGIN.
}
"""
    )
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-pthread",
            "-I" + str(root / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    path = tmp_path / "progress.jsonl"
    env = dict(os.environ, VIBEQC_DF_PROGRESS_TRACE=str(path))
    subprocess.run([str(executable)], env=env, check=True)
    journal = read_progress(path)
    assert not journal["complete"]
    assert [r["name"] for r in journal["pending"]] == [
        "endpoint",
        "raw_materialization",
    ]
    summary = summarize_progress(journal)
    assert summary["phases"]["host:setup"]["calls"] == 1
    assert [r["value"] for r in summary["observations"]] == [768, "generated"]
    with path.open("ab") as file:
        file.write(b'{"schema":')
    assert read_progress(path)["truncated_tail"]


def test_progress_rejects_capture_as_execution_and_scope_corruption(
    tmp_path: typing.Any,
) -> None:
    path = tmp_path / "trace.jsonl"
    begin = {
        "schema": "vibeqc.df_progress",
        "version": 1,
        "id": 0,
        "parent": -1,
        "time_ns": 10,
        "elapsed_ms": 0,
        "event": "BEGIN",
        "status": "started",
        "execution": "graph_capture",
        "name": "ri_k",
    }
    end = dict(begin, time_ns=20, event="END", status="graph_constructed")

    def write(*rows: typing.Any) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    write(begin, end)
    assert read_progress(path)["complete"]
    write(begin, dict(end, status="stream_complete"))
    with pytest.raises(ValueError, match="graph construction"):
        read_progress(path)
    write(begin, begin)
    with pytest.raises(ValueError, match="duplicate"):
        read_progress(path)
    write(begin, dict(end, name="another"))
    with pytest.raises(ValueError, match="identity changed"):
        read_progress(path)


def test_killed_native_probe_retains_completed_rows(tmp_path: typing.Any) -> None:
    from benchmarks.issue308_stage_probe import completed_native_rows

    path = tmp_path / "native.jsonl"
    path.write_bytes(b'{"operation":"setup","seconds":103}\n{"operation":')
    rows, truncated = completed_native_rows(path)
    assert rows == [{"operation": "setup", "seconds": 103}]
    assert truncated
