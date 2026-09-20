from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from vibeqc.autotune import _source_identity_paths


def _write_manifest(root: Path, payload: dict[str, object]) -> None:
    manifest = root / "cmake/VibeQCSourceIdentity.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload), encoding="utf-8")


def _configure_manifest(root: Path) -> subprocess.CompletedProcess[str]:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for the cross-consumer manifest gate")
    helper = Path(__file__).resolve().parents[2] / "cmake/VibeQCSourceIdentity.cmake"
    shutil.copyfile(helper, root / "cmake/VibeQCSourceIdentity.cmake")
    (root / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(source_identity_test NONE)\n"
        "include(cmake/VibeQCSourceIdentity.cmake)\n"
        "vibeqc_collect_source_identity_inputs(inputs)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/inputs.txt" "${inputs}")\n',
        encoding="utf-8",
    )
    return subprocess.run(
        [cmake, "-S", str(root), "-B", str(root / "build")],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_source_identity_manifest_expands_recursive_and_explicit_inputs(
    tmp_path: Path,
) -> None:
    (tmp_path / "src/nested").mkdir(parents=True)
    (tmp_path / "src/nested/kernel.cpp").write_text("kernel\n", encoding="utf-8")
    (tmp_path / "src/nested/ignored.txt").write_text("ignored\n", encoding="utf-8")
    (tmp_path / "CMakeLists.txt").write_text("project(test)\n", encoding="utf-8")
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "recursive_groups": [{"root": "src", "patterns": ["*.cpp"]}],
            "files": ["CMakeLists.txt"],
        },
    )

    assert {
        path.relative_to(tmp_path).as_posix()
        for path in _source_identity_paths(tmp_path)
    } == {
        "CMakeLists.txt",
        "cmake/VibeQCSourceIdentity.json",
        "src/nested/kernel.cpp",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": 1,
            "recursive_groups": [{"root": "../outside", "patterns": ["*"]}],
            "files": [],
        },
        {
            "schema_version": 1,
            "recursive_groups": [],
            "files": ["../outside"],
        },
    ],
)
def test_source_identity_manifest_rejects_parent_traversal(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    _write_manifest(tmp_path, payload)
    with pytest.raises(ValueError, match="stay inside"):
        _source_identity_paths(tmp_path)


def test_source_identity_manifest_fails_closed_on_missing_explicit_file(
    tmp_path: Path,
) -> None:
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "recursive_groups": [],
            "files": ["missing.txt"],
        },
    )
    with pytest.raises(FileNotFoundError, match="missing.txt"):
        _source_identity_paths(tmp_path)


@pytest.mark.parametrize("pattern", ["", "../outside/*", "/absolute/*", 1, True])
def test_cmake_and_python_reject_unsafe_patterns(
    tmp_path: Path,
    pattern: object,
) -> None:
    (tmp_path / "src").mkdir()
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "recursive_groups": [{"root": "src", "patterns": [pattern]}],
            "files": [],
        },
    )
    with pytest.raises(ValueError):
        _source_identity_paths(tmp_path)
    result = _configure_manifest(tmp_path)
    assert result.returncode != 0
    assert "Unsafe source identity pattern" in result.stderr


def test_cmake_and_python_expand_the_same_valid_inventory(tmp_path: Path) -> None:
    (tmp_path / "src/nested").mkdir(parents=True)
    (tmp_path / "src/nested/a.cpp").write_text("a\n", encoding="utf-8")
    (tmp_path / "src/b.cpp").write_text("b\n", encoding="utf-8")
    (tmp_path / "src/ignored.txt").write_text("ignored\n", encoding="utf-8")
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "recursive_groups": [{"root": "src", "patterns": ["*.cpp", "*.cpp"]}],
            "files": ["CMakeLists.txt", "src/b.cpp"],
        },
    )
    result = _configure_manifest(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = [
        path.relative_to(tmp_path).as_posix()
        for path in _source_identity_paths(tmp_path)
    ]
    assert (tmp_path / "build/inputs.txt").read_text().split(";") == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 1, "recursive_groups": {}, "files": []},
        {"schema_version": 1, "recursive_groups": [], "files": {}},
        {"schema_version": 1, "recursive_groups": ["invalid"], "files": []},
    ],
)
def test_manifest_container_type_errors(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    _write_manifest(tmp_path, payload)
    with pytest.raises(TypeError):
        _source_identity_paths(tmp_path)
