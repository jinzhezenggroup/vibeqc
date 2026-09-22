from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3/gfn1.json"
RAW = ROOT / "upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3/gfn1.toml"
MANIFEST = (
    ROOT
    / "upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3/gfn1_manifest.json"
)
HEADER = ROOT / "src/xtb/native/data/parameters/gfn1.hpp"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gfn1_snapshots_match_audited_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert _sha256(RAW) == manifest["outputs"]["gfn1.toml"]["sha256"]
    assert _sha256(SOURCE) == manifest["outputs"]["gfn1.json"]["sha256"]
    # The sole native adaptation is the namespace; the upstream digest remains
    # an independent byte-for-byte gate on all scientific parameters/schema.
    upstream_header = HEADER.read_bytes().replace(
        b"namespace vibeqc::xtb::parameters::gfn1",
        b"namespace xtbloom::parameters::gfn1",
    )
    assert (
        hashlib.sha256(upstream_header).hexdigest()
        == manifest["outputs"]["gfn1.hpp"]["sha256"]
    )
    assert manifest["source"]["revision"] == "fa8a4416e8fe093d0075bc10ac875494c2a449a9"
    assert manifest["mctc"]["revision"] == "e9de066d89f250d1cfb6de3a33f0c27c0e2f855d"


def test_gfn1_generator_reproduces_audited_header() -> None:
    subprocess.run(
        [sys.executable, "tools/parameters/generate_gfn1.py", "--check"],
        cwd=ROOT,
        check=True,
    )


def test_gfn1_generated_header_shape_is_explicit() -> None:
    text = HEADER.read_text(encoding="utf-8")
    assert "kElementCount = 86u" in text
    assert "kShellCount = 237u" in text
    assert "std::array<PairScaleOverride, 869u>" in text
    assert "double halogen_damping;" in text
    assert "double coordination_cutoff_bohr;" in text
