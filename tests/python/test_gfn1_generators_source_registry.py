"""Common source-registry ownership for GFN1 generated products."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import source_registry
from tools.parameters import generate_gfn1, generate_gfn1_geometry

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("module", "product_id"),
    (
        (generate_gfn1, "gfn1-parameter-header"),
        (generate_gfn1_geometry, "gfn1-geometry-python-data"),
    ),
)
def test_gfn1_generators_bind_remote_registered_source(
    module: object, product_id: str
) -> None:
    registry = source_registry._load(source_registry.REGISTRY)
    source = registry["sources"][module.SOURCE_ID]
    product = registry["products"][product_id]
    assert source["kind"] == "remote-file-set"
    assert "local_root" not in source
    assert product["inputs"] == [module.SOURCE_ID]
    assert set(module._REQUIRED_SOURCE_FILES) <= set(source["files"])


@pytest.mark.parametrize("module", (generate_gfn1, generate_gfn1_geometry))
def test_gfn1_generators_require_registered_manifest_file(
    monkeypatch: pytest.MonkeyPatch, module: object
) -> None:
    monkeypatch.setattr(
        source_registry,
        "read_source_texts",
        lambda *_args, **_kwargs: {"gfn1.json": "{}"},
    )
    with pytest.raises(source_registry.SourceRegistryError, match="required files"):
        module.load_registered_inputs()


@pytest.mark.parametrize("module", (generate_gfn1, generate_gfn1_geometry))
def test_gfn1_generators_propagate_registry_integrity_failures(
    monkeypatch: pytest.MonkeyPatch, module: object
) -> None:
    def fail(*_args: object, **_kwargs: object) -> dict[str, str]:
        raise source_registry.SourceRegistryError("injected digest mismatch")

    monkeypatch.setattr(source_registry, "read_source_texts", fail)
    with pytest.raises(source_registry.SourceRegistryError, match="digest mismatch"):
        module.load_registered_inputs()


def test_remote_sources_keep_exact_upstream_urls() -> None:
    registry = json.loads((ROOT / "upstream/manifest.json").read_text())
    for source_id in ("xtbloom-gfn1-d3", "xtbloom-gfn1-parameters"):
        for item in registry["sources"][source_id]["files"].values():
            assert item["url"].startswith("https://raw.githubusercontent.com/")
