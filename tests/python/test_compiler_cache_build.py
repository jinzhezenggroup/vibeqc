"""Check compiler-cache launchers preserve portable build-directory keys."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _configure(binary: Path, *extra: str) -> None:
    subprocess.run(
        [
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(binary),
            "-G",
            "Ninja",
            "-DVIBEQC_ENABLE_CUDA=OFF",
            "-DVIBEQC_BUILD_TESTS=OFF",
            *extra,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _vibeqc_commands(binary: Path) -> str:
    return subprocess.check_output(
        ["ninja", "-C", str(binary), "-t", "commands", "vibeqc"], text=True
    )


def test_ccache_launcher_normalizes_binary_directory(tmp_path: Path) -> None:
    """Generated objects should reuse ccache entries across build directories."""
    if shutil.which("ccache") is None:
        pytest.skip("ccache is not installed")

    binary = tmp_path / "build"
    _configure(binary, "-DVIBEQC_COMPILER_CACHE=ccache")
    commands = _vibeqc_commands(binary)

    assert f"CCACHE_BASEDIR={binary}" in commands
    assert "ccache" in commands


def test_explicit_compiler_launcher_remains_authoritative(tmp_path: Path) -> None:
    """Do not replace a caller-owned launcher with the project cache wrapper."""
    if shutil.which("ccache") is None:
        pytest.skip("ccache is not installed")

    launcher = tmp_path / "launcher"
    launcher.write_text('#!/bin/sh\nexec "$@"\n')
    launcher.chmod(0o755)
    binary = tmp_path / "build"
    _configure(
        binary,
        "-DVIBEQC_COMPILER_CACHE=ccache",
        f"-DCMAKE_CXX_COMPILER_LAUNCHER={launcher}",
    )
    commands = _vibeqc_commands(binary)

    assert str(launcher) in commands
    assert "CCACHE_BASEDIR=" not in commands
