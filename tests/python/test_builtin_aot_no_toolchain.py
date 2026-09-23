"""Built-in execution must stay independent of local compiler toolchains."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_builtin_rhf_runs_without_toolchain_activation(tmp_path: Path) -> None:
    """A qualified AOT built-in must not probe or launch a compiler."""

    root = Path(__file__).resolve().parents[2]
    guard = tmp_path / "sitecustomize.py"
    guard.write_text(
        """
import os
import shutil
import subprocess
from pathlib import Path

_TOOLCHAIN_NAMES = frozenset(
    {
        "cc",
        "c++",
        "gcc",
        "g++",
        "clang",
        "clang++",
        "clang-cl",
        "cl",
        "cl.exe",
        "nvcc",
        "cmake",
        "ninja",
    }
)
_real_which = shutil.which


def _which_without_toolchain(command, *args, **kwargs):
    name = Path(os.fsdecode(command)).name.lower()
    if name in _TOOLCHAIN_NAMES:
        return None
    return _real_which(command, *args, **kwargs)


def _forbid_subprocess(*args, **kwargs):
    command = args[0] if args else kwargs.get("args")
    raise AssertionError(
        f"ordinary built-in execution attempted to launch a subprocess: {command!r}"
    )


shutil.which = _which_without_toolchain
subprocess.Popen = _forbid_subprocess
""".lstrip(),
        encoding="utf-8",
    )

    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = [str(tmp_path), str(root / "python")]
    if existing_pythonpath:
        pythonpath.append(existing_pythonpath)
    missing = tmp_path / "missing-toolchain"

    code = """
from vibeqc import Calculator

atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
result = Calculator(method="rhf", basis="sto-3g", device="cpu").singlepoint(atoms)
assert result.converged
assert result.executed_backend == "cpu_reference"
assert abs(result.energy - (-1.11671432506255)) < 2.0e-9
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "CC": str(missing / "cc"),
            "CXX": str(missing / "c++"),
            "NVCC": str(missing / "nvcc"),
            "CMAKE_CUDA_COMPILER": str(missing / "nvcc"),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
