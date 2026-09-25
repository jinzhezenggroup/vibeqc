"""Remote source ownership and checked-in compact D3 production data."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from vibeqc_compiler.common.d3_data import load_d3_production_data

from tools import source_registry
from tools.vibeqc_d3 import generate_compact_data, generate_native_data


def test_checked_in_compact_d3_product_matches_registry() -> None:
    registry = source_registry._load(source_registry.REGISTRY)
    product = registry["products"][generate_compact_data.PRODUCT_ID]
    output = generate_compact_data.DEFAULT_OUTPUT
    assert (
        hashlib.sha256(output.read_bytes()).hexdigest()
        == product["outputs"]["data/parameters/d3_production.bin"]
    )
    data = load_d3_production_data(output)
    assert (
        data.table_sha256
        == "9ff932ea598f690c1fb599a67762060ba1907102d5ec132164f2a7e8886cd22e"
    )
    assert (
        data.radii_sha256
        == "92b32fada844a337204b84f2d961473bad5737240765eb8d0727a62827de5111"
    )
    assert len(data.pairs) == 3741
    assert len(data.c6) == 28455


def test_native_d3_generator_consumes_only_compact_product() -> None:
    implementation = Path(generate_native_data.__file__).read_text(encoding="utf-8")
    assert "source_registry" not in implementation
    assert "upstream/xtbloom/" not in implementation
    assert "d3_production.bin" in implementation
    rendered = generate_native_data.render()
    assert "kReferenceC6" in rendered
    assert "kPairs" in rendered


def test_compact_generator_binds_remote_sources() -> None:
    registry = source_registry._load(source_registry.REGISTRY)
    product = registry["products"][generate_compact_data.PRODUCT_ID]
    assert product["inputs"] == list(generate_compact_data.PRODUCT_INPUTS)
    for source_id in generate_compact_data.PRODUCT_INPUTS:
        source = registry["sources"][source_id]
        assert source["kind"] == "remote-file-set"
        assert "local_root" not in source


def test_compact_generator_propagates_registry_integrity_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> dict[str, str]:
        raise source_registry.SourceRegistryError("injected digest mismatch")

    monkeypatch.setattr(source_registry, "read_source_texts", fail)
    with pytest.raises(source_registry.SourceRegistryError, match="digest mismatch"):
        generate_compact_data.render()
