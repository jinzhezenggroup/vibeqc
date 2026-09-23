"""Common source-registry ownership for the repository-only D3 reference."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools import source_registry
from tools.vibeqc_d3 import reference


def test_d3_reference_reads_exact_sources_from_common_registry() -> None:
    registry = source_registry._load(source_registry.REGISTRY)
    source = registry["sources"][reference._SOURCE_ID]
    texts = reference._registered_source_texts()

    assert set(texts) == reference._REQUIRED_SOURCE_FILES
    for name, text in texts.items():
        assert (
            hashlib.sha256(text.encode("utf-8")).hexdigest()
            == source["files"][name]["sha256"]
        )

    implementation = Path(reference.__file__).read_text(encoding="utf-8")
    assert source["revision"] not in implementation
    assert "upstream/xtbloom/" not in implementation


def test_d3_reference_fails_closed_when_registered_source_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = source_registry._load(source_registry.REGISTRY)
    broken = {**registry, "sources": dict(registry["sources"])}
    broken["sources"].pop(reference._SOURCE_ID)
    monkeypatch.setattr(source_registry, "_load", lambda _path: broken)

    with pytest.raises(source_registry.SourceRegistryError, match="missing"):
        reference._registered_source_texts()


def test_d3_reference_requires_both_registered_input_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        source_registry,
        "read_source_texts",
        lambda _source_id, _source: {"gfn1.json": "{}"},
    )

    with pytest.raises(source_registry.SourceRegistryError, match="required files"):
        reference._registered_source_texts()
