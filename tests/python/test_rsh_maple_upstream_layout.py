"""Production adapters follow canonical upstream sources without stale fallback."""

import shutil
from pathlib import Path

import pytest
from vibeqc_compiler.xc import rsh_maple as adapter


def test_canonical_upstream_directory_preserves_imported_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factories = [(adapter._lyp_module, ())]
    for factory, _ in factories:
        factory.cache_clear()
    before = [factory(*args).transitive_sha256 for factory, args in factories]
    root = adapter._libxc_root()
    canonical = tmp_path / "upstream/libxc/7.0.0"
    shutil.copytree(root, canonical)
    legacy = tmp_path / "external/libxc-7.0.0"
    legacy.mkdir(parents=True)
    monkeypatch.setattr(
        adapter,
        "asset_path",
        lambda path: canonical if path == "upstream/libxc/7.0.0" else legacy,
    )
    for factory, _ in factories:
        factory.cache_clear()
    try:
        assert adapter._libxc_root() == canonical
        assert [
            factory(*args).transitive_sha256 for factory, args in factories
        ] == before
    finally:
        for factory, _ in factories:
            factory.cache_clear()
