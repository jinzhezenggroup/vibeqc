"""Common source-registry ownership for GFN1 generated products."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools import source_registry
from tools.parameters import generate_gfn1, generate_gfn1_geometry


@pytest.mark.parametrize(
    ("module", "output"),
    (
        (generate_gfn1, generate_gfn1.DEFAULT_OUTPUT),
        (generate_gfn1_geometry, generate_gfn1_geometry.DEFAULT_OUTPUT),
    ),
)
def test_gfn1_generators_use_registered_source_and_preserve_product_bytes(
    module: object, output: Path
) -> None:
    source_bytes, manifest = module.load_registered_inputs()
    registry = source_registry._load(source_registry.REGISTRY)
    source = registry["sources"][module.SOURCE_ID]
    implementation = Path(module.__file__).read_text(encoding="utf-8")

    assert (
        hashlib.sha256(source_bytes).hexdigest()
        == source["files"]["gfn1.json"]["sha256"]
    )
    assert source["revision"] not in implementation
    assert "upstream/xtbloom/" not in implementation
    if module is generate_gfn1:
        rendered = module.load_and_render(source_bytes, manifest)
    else:
        rendered = module.render(source_bytes, manifest)
    assert rendered == output.read_bytes()


@pytest.mark.parametrize("module", (generate_gfn1, generate_gfn1_geometry))
def test_gfn1_generators_require_registered_manifest_file(
    monkeypatch: pytest.MonkeyPatch, module: object
) -> None:
    original = source_registry.read_source_texts

    def read_without_manifest(
        source_id: str, source: dict, **kwargs: object
    ) -> dict[str, str]:
        texts = original(source_id, source, **kwargs)
        texts.pop("gfn1_manifest.json")
        return texts

    monkeypatch.setattr(source_registry, "read_source_texts", read_without_manifest)
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
