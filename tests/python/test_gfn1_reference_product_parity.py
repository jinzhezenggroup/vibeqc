"""Remote reference storage must not remove byte-exact GFN1 regeneration gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import source_registry
from tools.parameters import generate_gfn1, generate_gfn1_geometry

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("module", "output"),
    (
        (generate_gfn1, generate_gfn1.DEFAULT_OUTPUT),
        (generate_gfn1_geometry, generate_gfn1_geometry.DEFAULT_OUTPUT),
    ),
)
def test_registered_upstream_inputs_reproduce_current_products(
    module: object, output: Path
) -> None:
    source_bytes, manifest = module.load_registered_inputs()
    registry = source_registry._load(source_registry.REGISTRY)
    source = registry["sources"][module.SOURCE_ID]
    assert (
        hashlib.sha256(source_bytes).hexdigest()
        == source["files"]["gfn1.json"]["sha256"]
    )
    if module is generate_gfn1:
        rendered = module.load_and_render(source_bytes, manifest)
    else:
        rendered = module.render(source_bytes, manifest)
    assert rendered == output.read_bytes()


def test_gfn1_products_retain_the_independent_upstream_audit() -> None:
    registry = source_registry._load(source_registry.REGISTRY)
    source_id = "xtbloom-gfn1-parameters"
    texts = source_registry.read_source_texts(source_id, registry["sources"][source_id])
    manifest = json.loads(texts["gfn1_manifest.json"])
    for name in ("gfn1.toml", "gfn1.json"):
        assert (
            hashlib.sha256(texts[name].encode("utf-8")).hexdigest()
            == manifest["outputs"][name]["sha256"]
        )
    # The native adaptation changes only the namespace, not parameter bytes.
    header = (ROOT / "src/xtb/native/data/parameters/gfn1.hpp").read_bytes()
    upstream_header = header.replace(
        b"namespace vibeqc::xtb::parameters::gfn1",
        b"namespace xtbloom::parameters::gfn1",
    )
    assert (
        hashlib.sha256(upstream_header).hexdigest()
        == manifest["outputs"]["gfn1.hpp"]["sha256"]
    )
    assert manifest["source"]["revision"] == "fa8a4416e8fe093d0075bc10ac875494c2a449a9"
    assert manifest["mctc"]["revision"] == "e9de066d89f250d1cfb6de3a33f0c27c0e2f855d"
