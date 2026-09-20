from __future__ import annotations

import json
from pathlib import Path

import pytest
from vibeqc.autotune import _source_identity_paths


def _write_manifest(root: Path, payload: dict[str, object]) -> None:
    manifest = root / "cmake/VibeQCSourceIdentity.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload), encoding="utf-8")


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
