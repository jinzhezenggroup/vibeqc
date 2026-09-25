#!/usr/bin/env python3
"""Regenerate compact D3 production data from pinned remote source-registry inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import source_registry

PRODUCT_ID = "d3-production-data"
PRODUCT_INPUTS = ("xtbloom-gfn1-d3", "xtbloom-gfn1-parameters")
D3_SOURCE_ID, MODEL_SOURCE_ID = PRODUCT_INPUTS
MANIFEST = ROOT / "manifests/xtbloom-d3.json"
DEFAULT_OUTPUT = ROOT / "data/parameters/d3_production.bin"
MAGIC = b"VQD3BIN1"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def render(
    *,
    registry_path: Path = source_registry.REGISTRY,
    cache_root: Path = source_registry.DEFAULT_CACHE,
) -> bytes:
    r = source_registry.load_product_sources(
        PRODUCT_ID,
        generator=Path(__file__),
        expected_inputs=PRODUCT_INPUTS,
        expected_canonical_inputs=(MANIFEST.relative_to(ROOT).as_posix(),),
        registry_path=registry_path,
    )
    a, b = r[D3_SOURCE_ID], r[MODEL_SOURCE_ID]
    if a["revision"] != b["revision"]:
        raise source_registry.SourceRegistryError(
            "D3 and GFN1 source owners must pin the same upstream revision"
        )
    dt = source_registry.read_source_texts(D3_SOURCE_ID, a, cache_root=cache_root)
    gt = source_registry.read_source_texts(MODEL_SOURCE_ID, b, cache_root=cache_root)
    dr = dt["gfn1_d3.json"].encode()
    gr = gt["gfn1.json"].encode()
    m = json.loads(MANIFEST.read_text())
    if (
        _sha(dr) != m["data"]["gfn1_d3.json"]
        or _sha(gr) != m["sources"]["data/parameters/gfn1.json"]["sha256"]
    ):
        raise ValueError("registered D3/GFN1 source digest mismatch")
    d = json.loads(dr)
    g = json.loads(gr)
    radii = [float(x["covalent_radius_bohr"]) for x in g["elements"]]
    if (
        _sha((json.dumps(radii, indent=2) + "\n").encode())
        != m["data"]["covalent_radii.json"]
    ):
        raise ValueError("derived covalent-radii digest mismatch")
    out = bytearray(
        MAGIC
        + a["revision"].encode()
        + bytes.fromhex(m["data"]["gfn1_d3.json"])
        + bytes.fromhex(m["data"]["covalent_radii.json"])
        + struct.pack("<7I", 86, 3741, 237, 28455, 86, 3741, 86)
    )
    for x in d["elements"]:
        out += struct.pack("<IB", int(x["reference_offset"]), int(x["reference_count"]))
    for x in d["pair_records"]:
        out += struct.pack(
            "<IBB",
            int(x["c6_offset"]),
            int(x["first_reference_count"]),
            int(x["second_reference_count"]),
        )
    for values in (
        d["coordination_numbers"],
        d["c6"],
        d["r4r2"],
        d["vdw_radii"],
        radii,
    ):
        out += struct.pack(f"<{len(values)}d", *(float(v) for v in values))
    return bytes(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, default=source_registry.REGISTRY)
    p.add_argument("--cache-root", type=Path, default=source_registry.DEFAULT_CACHE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    v = render(registry_path=a.registry, cache_root=a.cache_root)
    if a.check:
        if not a.output.is_file() or a.output.read_bytes() != v:
            raise SystemExit(f"{a.output} is stale; regenerate it")
        return 0
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_bytes(v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
