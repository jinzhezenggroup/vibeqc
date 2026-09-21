#!/usr/bin/env python3
"""Generate the compiler-owned GFN1 geometry parameter subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "sources/xtb/gfn1/gfn1.json"
DEFAULT_MANIFEST = ROOT / "sources/xtb/gfn1/gfn1_manifest.json"
DEFAULT_OUTPUT = ROOT / "python/vibeqc_compiler/geometry/_gfn1_data.py"


class Gfn1GeometryDataError(ValueError):
    """Pinned GFN1 geometry inputs no longer match the audited contract."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Gfn1GeometryDataError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Gfn1GeometryDataError(f"{label} must be finite")
    return result


def _render_float(value: Any, label: str) -> str:
    return repr(_finite(value, label))


def render(source_bytes: bytes, manifest: dict[str, Any]) -> bytes:
    parameters = json.loads(source_bytes)
    expected_source = manifest["outputs"]["gfn1.json"]["sha256"]
    if _sha256(source_bytes) != expected_source:
        raise Gfn1GeometryDataError("GFN1 JSON differs from the audited manifest")
    if parameters.get("method") != "gfn1-xtb" or parameters.get("schema_version") != 2:
        raise Gfn1GeometryDataError("unsupported GFN1 normalized schema")

    coordination = parameters["coordination_number"]
    if (
        coordination["model"] != "exp"
        or coordination["cutoff_bohr"] != 25.0
        or coordination["steepness"] != 16.0
        or coordination["coincident_distance_squared_cutoff_bohr2"] != 1.0e-12
        or coordination["cutoff_inclusive"] is not True
        or coordination["coincident_cutoff_inclusive"] is not True
        or coordination["maximum_cn_cutoff"] is not None
    ):
        raise Gfn1GeometryDataError("GFN1 coordination semantics changed")

    repulsion = parameters["repulsion"]
    if repulsion != {"kexp": 1.5, "klight": 1.5}:
        raise Gfn1GeometryDataError("GFN1 repulsion semantics changed")

    halogen = parameters["halogen"]
    if halogen != {"damping": 0.44, "radius_scale": 1.3}:
        raise Gfn1GeometryDataError("GFN1 halogen semantics changed")

    elements = parameters["elements"]
    if not isinstance(elements, list) or len(elements) != 86:
        raise Gfn1GeometryDataError("GFN1 geometry table must contain H through Rn")

    rows = []
    for atomic_number, element in enumerate(elements, 1):
        if element["atomic_number"] != atomic_number:
            raise Gfn1GeometryDataError("GFN1 elements are not in atomic-number order")
        row = (
            atomic_number,
            _finite(element["covalent_radius_bohr"], "covalent_radius_bohr"),
            _finite(element["arep"], "arep"),
            _finite(element["zeff"], "zeff"),
            _finite(element["atomic_radius_bohr"], "atomic_radius_bohr"),
            _finite(element["xbond"], "xbond"),
        )
        if any(value <= 0 for value in row[1:5]):
            raise Gfn1GeometryDataError(
                f"GFN1 element {atomic_number} has a nonpositive geometry parameter"
            )
        if row[5] < 0:
            raise Gfn1GeometryDataError(
                f"GFN1 element {atomic_number} has a negative halogen bond strength"
            )
        rows.append(row)

    lines = [
        '"""Generated GFN1 geometry data; do not edit by hand."""',
        "",
        "GFN1_PARAMETER_JSON_SHA256 = (",
        f'    "{expected_source}"',
        ")",
        f"GFN1_COORDINATION_STEEPNESS = {_render_float(coordination['steepness'], 'steepness')}",
        f"GFN1_CUTOFF_BOHR = {_render_float(coordination['cutoff_bohr'], 'cutoff')}",
        (
            "GFN1_MINIMUM_DISTANCE_SQUARED_BOHR2 = "
            + _render_float(
                coordination["coincident_distance_squared_cutoff_bohr2"],
                "minimum_distance_squared",
            )
        ),
        f"GFN1_REPULSION_KEXP = {_render_float(repulsion['kexp'], 'kexp')}",
        f"GFN1_REPULSION_KLIGHT = {_render_float(repulsion['klight'], 'klight')}",
        f"GFN1_HALOGEN_DAMPING = {_render_float(halogen['damping'], 'halogen.damping')}",
        (
            "GFN1_HALOGEN_RADIUS_SCALE = "
            + _render_float(halogen["radius_scale"], "halogen.radius_scale")
        ),
        "",
        "GFN1_GEOMETRY_ELEMENT_ROWS = (",
    ]
    for row in rows:
        lines.append(
            "    ("
            + ", ".join((str(row[0]),) + tuple(repr(float(value)) for value in row[1:]))
            + "),"
        )
    lines.extend((")", ""))
    return "\n".join(lines).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rendered = render(
        args.source.read_bytes(),
        json.loads(args.manifest.read_text(encoding="utf-8")),
    )
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != rendered:
            raise Gfn1GeometryDataError(
                f"generated GFN1 geometry data is stale: {args.output}"
            )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
