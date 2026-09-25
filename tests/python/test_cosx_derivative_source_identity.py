"""Native COSX wrapper edits must invalidate build/tuning source identity."""

import json
from pathlib import Path

import pytest
from vibeqc.autotune import _source_identity_paths, source_identity

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = Path("tools/generate_cosx_derivative_native.py")


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    # Populate the real inventory in an isolated metadata-only checkout. This
    # exercises the production expander/hash, not a second test-only hash.
    manifest = ROOT / "cmake/VibeQCSourceIdentity.json"
    payload = json.loads(manifest.read_text())
    for group in payload["recursive_groups"]:
        (tmp_path / group["root"]).mkdir(parents=True, exist_ok=True)
    for name in payload["files"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    (tmp_path / "cmake/VibeQCSourceIdentity.json").write_text(manifest.read_text())
    generator = tmp_path / GENERATOR
    generator.parent.mkdir(parents=True, exist_ok=True)
    generator.write_text("# original scalar wrapper\n")
    return tmp_path


def test_cosx_generator_is_a_source_identity_input(checkout: Path) -> None:
    assert checkout / GENERATOR in _source_identity_paths(checkout)


def test_cosx_wrapper_only_change_invalidates_source_identity(checkout: Path) -> None:
    before = source_identity(checkout)
    (checkout / GENERATOR).write_text("# changed scalar wrapper\n")
    assert source_identity(checkout) != before
