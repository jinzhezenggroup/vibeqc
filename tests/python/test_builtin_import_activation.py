"""Public runtime import must stay independent of compiler toolchain activation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_import_vibeqc_stays_toolchain_lazy(tmp_path: Path) -> None:
    """Importing the runtime must neither probe toolchains nor load JIT machinery."""

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


def _guarded_which(command, *args, **kwargs):
    name = Path(os.fsdecode(command)).name.lower()
    if name in _TOOLCHAIN_NAMES:
        raise AssertionError(
            f"importing vibeqc probed local compiler toolchain executable {name!r}"
        )
    return _real_which(command, *args, **kwargs)


def _forbid_subprocess(*args, **kwargs):
    command = args[0] if args else kwargs.get("args")
    raise AssertionError(
        f"importing vibeqc attempted to launch a subprocess: {command!r}"
    )


shutil.which = _guarded_which
subprocess.Popen = _forbid_subprocess
""".lstrip(),
        encoding="utf-8",
    )

    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = [str(tmp_path), str(root / "python")]
    if existing_pythonpath:
        pythonpath.append(existing_pythonpath)

    code = """
import sys

import vibeqc

forbidden_exact = {
    "vibeqc_compiler.common.cpp_adapter",
    "vibeqc_compiler.common.cuda_adapter",
    "vibeqc_compiler.common.native_runtime",
    "vibeqc_compiler.common.cuda_runtime",
}
forbidden_prefixes = (
    "vibeqc._stationary_cpu",
    "vibeqc._stationary_cuda",
)
forbidden = sorted(
    name
    for name in sys.modules
    if name in forbidden_exact or name.startswith(forbidden_prefixes)
)
assert not forbidden, (
    "ordinary runtime import activated compiler/JIT machinery: "
    + ", ".join(forbidden)
)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(pythonpath),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
