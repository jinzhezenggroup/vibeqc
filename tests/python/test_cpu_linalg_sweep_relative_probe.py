"""An explicitly selected local executable must not be searched only on PATH."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_relative_probe_executes_from_calling_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    source, probe = tmp_path / "local.cpp", tmp_path / "local-probe"
    source.write_text(
        '#include <iostream>\nint main() { std::cerr << "local probe executed"; return 1; }\n',
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, str(source), "-o", str(probe)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    spec = importlib.util.spec_from_file_location(
        "cpu_sweep_relative", ROOT / "benchmarks/cpu_linalg_sweep.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="local probe executed"):
        module.run_sweep(Path("./local-probe"), sizes=(16,), providers=("scalar",))
