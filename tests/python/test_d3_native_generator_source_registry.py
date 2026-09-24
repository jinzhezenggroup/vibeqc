"""Common source-registry ownership for the production D3 table generator."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from tools import source_registry
from tools.vibeqc_d3 import generate_native_data

_SOURCE_OWNERS = {
    "gfn1_d3.json": "xtbloom-gfn1-d3",
    "gfn1.json": "xtbloom-gfn1-parameters",
}


def test_d3_native_generator_reads_exact_sources_from_common_registry() -> None:
    registry = source_registry._load(source_registry.REGISTRY)
    texts, revision = generate_native_data._registered_source_texts()
    implementation = Path(generate_native_data.__file__).read_text(encoding="utf-8")

    assert set(texts) == generate_native_data._REQUIRED_SOURCE_FILES == set(_SOURCE_OWNERS)
    for name, text in texts.items():
        source = registry["sources"][_SOURCE_OWNERS[name]]
        assert source["revision"] == revision
        assert (
            hashlib.sha256(text.encode("utf-8")).hexdigest()
            == source["files"][name]["sha256"]
        )
        assert source["revision"] not in implementation
    assert "upstream/xtbloom/" not in implementation


def test_d3_native_generator_preserves_checked_in_product() -> None:
    output = generate_native_data.DEFAULT_OUTPUT.read_text(encoding="utf-8")
    assert generate_native_data.render() == output


@pytest.mark.parametrize("source_id", _SOURCE_OWNERS.values())
def test_d3_native_generator_fails_closed_when_registered_source_is_missing(
    monkeypatch: pytest.MonkeyPatch, source_id: str
) -> None:
    registry = source_registry._load(source_registry.REGISTRY)
    broken = copy.deepcopy(registry)
    broken["sources"].pop(source_id)
    monkeypatch.setattr(source_registry, "_load", lambda _path: broken)

    with pytest.raises(source_registry.SourceRegistryError, match="missing"):
        generate_native_data._registered_source_texts()


def test_d3_native_generator_rejects_split_upstream_revisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = source_registry._load(source_registry.REGISTRY)
    broken = copy.deepcopy(registry)
    broken["sources"][generate_native_data._MODEL_SOURCE_ID]["revision"] = "deadbeef"
    monkeypatch.setattr(source_registry, "_load", lambda _path: broken)

    def read(source_id: str, _source: object) -> dict[str, str]:
        return {
            name: "{}" for name, owner in _SOURCE_OWNERS.items() if owner == source_id
        }

    monkeypatch.setattr(source_registry, "read_source_texts", read)
    with pytest.raises(source_registry.SourceRegistryError, match="same upstream revision"):
        generate_native_data._registered_source_texts()


@pytest.mark.parametrize("missing", _SOURCE_OWNERS)
def test_d3_native_generator_requires_both_registered_input_files(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    def read(source_id: str, _source: object) -> dict[str, str]:
        return {
            name: "{}"
            for name, owner in _SOURCE_OWNERS.items()
            if owner == source_id and name != missing
        }

    monkeypatch.setattr(source_registry, "read_source_texts", read)
    with pytest.raises(source_registry.SourceRegistryError, match="required files"):
        generate_native_data._registered_source_texts()


@pytest.mark.parametrize("failed_owner", _SOURCE_OWNERS.values())
def test_d3_native_generator_propagates_registered_integrity_failure(
    monkeypatch: pytest.MonkeyPatch, failed_owner: str
) -> None:
    def read(source_id: str, _source: object) -> dict[str, str]:
        if source_id == failed_owner:
            raise source_registry.SourceRegistryError("injected digest mismatch")
        return {
            name: "{}" for name, owner in _SOURCE_OWNERS.items() if owner == source_id
        }

    monkeypatch.setattr(source_registry, "read_source_texts", read)
    with pytest.raises(source_registry.SourceRegistryError, match="digest mismatch"):
        generate_native_data._registered_source_texts()
