from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "src/xtb/native/data/parameters/gfn1.hpp"
REGISTRY = ROOT / "upstream/manifest.json"


def test_gfn1_generated_header_matches_registered_product() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    expected = registry["products"]["gfn1-parameter-header"]["outputs"][
        "src/xtb/native/data/parameters/gfn1.hpp"
    ]
    assert hashlib.sha256(HEADER.read_bytes()).hexdigest() == expected


def test_gfn1_remote_sources_are_not_checked_in_vendor_data() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    for source_id in ("xtbloom-gfn1-parameters", "xtbloom-gfn1-d3"):
        source = registry["sources"][source_id]
        assert source["kind"] == "remote-file-set"
        assert "local_root" not in source


def test_gfn1_generated_header_shape_is_explicit() -> None:
    text = HEADER.read_text(encoding="utf-8")
    assert "kElementCount = 86u" in text
    assert "kShellCount = 237u" in text
    assert "std::array<PairScaleOverride, 869u>" in text
    assert "double halogen_damping;" in text
    assert "double coordination_cutoff_bohr;" in text
