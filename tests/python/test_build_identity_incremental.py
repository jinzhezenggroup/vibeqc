"""Exercise the production identity build graph across source inventory changes."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from vibeqc.autotune import source_identity

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("generator", ["Ninja", "Unix Makefiles"])
def test_source_identity_tracks_incremental_inventory_changes(
    tmp_path: Path, generator: str
) -> None:
    """Membership edits must refresh identity without penalizing content-only edits."""
    build_tool = "ninja" if generator == "Ninja" else "make"
    if not shutil.which("cmake") or not shutil.which(build_tool):
        pytest.skip(f"CMake and {build_tool} are required")
    if generator == "Unix Makefiles" and sys.platform == "win32":
        pytest.skip("Unix Makefiles require a Unix build environment")

    # Use the actual top-level registration and manifest in an isolated checkout.
    # Native test/benchmark assets are unnecessary for this codegen-only target.
    source = tmp_path / "source"
    source.mkdir()
    for directory in ("cmake", "src", "include", "python", "tools"):
        shutil.copytree(
            ROOT / directory,
            source / directory,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    cmake = source / "CMakeLists.txt"
    cmake.write_text(
        (ROOT / "CMakeLists.txt").read_text()
        # Count configure passes without relying on generator-specific messages.
        + '\nfile(APPEND "${CMAKE_BINARY_DIR}/configure_runs.txt" "configure\\n")\n'
    )
    member = source / "python/vibeqc/_identity_incremental_test.py"
    member.write_text("VALUE = 1\n")
    build = tmp_path / "build"
    header = build / "generated/build_identity.hpp"
    configure_runs = build / "configure_runs.txt"

    def run(*arguments: str) -> None:
        result = subprocess.run(
            arguments, capture_output=True, text=True, timeout=60, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def rebuild() -> str:
        run("cmake", "--build", str(build), "--target", "vibeqc_build_identity_codegen")
        match = re.search(
            r'kVibeqcSourceIdentity = "([a-f0-9]{64})"', header.read_text()
        )
        assert match is not None
        identity = match.group(1)
        assert identity == source_identity(source)
        return identity

    configure = (
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        generator,
        "-DVIBEQC_ENABLE_CUDA=OFF",
        "-DVIBEQC_BUILD_TESTS=OFF",
        "-DVIBEQC_CPU_LINALG_PROVIDER=scalar",
        "-DVIBEQC_COMPILER_CACHE=off",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DPython3_EXECUTABLE={sys.executable}",
    )
    run(*configure)
    initial = rebuild()
    configured = configure_runs.read_text()

    member.write_text("VALUE = 200\n")
    edited = rebuild()
    assert edited != initial
    assert configure_runs.read_text() == configured

    # Touches and unchanged configure passes must not rewrite a valid header.
    header_mtime = header.stat().st_mtime_ns
    member.touch()
    assert rebuild() == edited
    assert header.stat().st_mtime_ns == header_mtime
    assert configure_runs.read_text() == configured
    run(*configure)
    assert rebuild() == edited
    assert header.stat().st_mtime_ns == header_mtime
    configured = configure_runs.read_text()

    member.unlink()
    deleted = rebuild()
    assert deleted != edited
    assert configure_runs.read_text() != configured
    configured = configure_runs.read_text()

    # Membership is a dependency even when a restored file predates the output.
    member.write_text("VALUE = 300\n")
    os.utime(member, (1, 1))
    added = rebuild()
    assert added != deleted
    assert configure_runs.read_text() != configured
    configured = configure_runs.read_text()
    member.rename(member.with_name("_identity_renamed_test.py"))
    renamed = rebuild()
    assert renamed != added
    assert configure_runs.read_text() != configured
    configured = configure_runs.read_text()

    # Changing the manifest must refresh the dependency graph: later edits of
    # a newly explicit input must be tracked without another configure pass.
    explicit = source / "tools/identity_incremental_test.txt"
    explicit.write_text("explicit input\n")
    manifest = source / "cmake/VibeQCSourceIdentity.json"
    payload = json.loads(manifest.read_text())
    payload["files"].append(explicit.relative_to(source).as_posix())
    manifest.write_text(json.dumps(payload))
    expanded = rebuild()
    assert expanded != renamed
    assert configure_runs.read_text() != configured
    configured = configure_runs.read_text()
    explicit.write_text("edited explicit input\n")
    assert rebuild() != expanded
    assert configure_runs.read_text() == configured
