"""Restore/check the small D3 data import from a pinned local xTBloom checkout."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[2] / "external/xtbloom-d3"
    manifest = json.loads((target / "manifest.json").read_text())
    sources = ("data/parameters/gfn1_d3.json", "data/parameters/gfn1.json")
    for name in sources:
        raw = (args.source / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["sources"][name]["sha256"]:
            raise ValueError(f"upstream data does not match pinned source: {name}")
    model = json.loads((args.source / sources[1]).read_text())
    if [e["atomic_number"] for e in model["elements"]] != list(range(1, 87)):
        raise ValueError("upstream element order changed")
    outputs = {
        "gfn1_d3.json": (args.source / sources[0]).read_bytes(),
        "covalent_radii.json": (
            json.dumps([e["covalent_radius_bohr"] for e in model["elements"]], indent=2)
            + "\n"
        ).encode(),
    }
    for name, raw in outputs.items():
        if hashlib.sha256(raw).hexdigest() != manifest["data"][name]:
            raise ValueError(f"derived data differs from audited import: {name}")
        if args.check:
            if (target / name).read_bytes() != raw:
                raise ValueError(
                    f"tracked data differs from reproducible import: {name}"
                )
        else:
            (target / name).write_bytes(raw)
    print("D3 source and data digests verified")


if __name__ == "__main__":
    main()
