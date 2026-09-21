#!/usr/bin/env python3
"""Build a reproducible bulk Graph catalog from the pinned full Libxc source tree.

Bootstrap explicitly with --archive /path/to/libxc-7.0.0.tar.gz. Regeneration and
--check are offline. The archive is an input, not a runtime dependency; only
Maple definitions, corresponding C parameter owners and provenance are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from vibeqc_compiler.xc import libxc_bulk, libxc_maple

from tools import source_registry
from tools.libxc_bulk_metadata import C_BINDING_SEMANTICS, extract_registrations

REVISION = "7.0.0"
ARCHIVE_URL = "https://gitlab.com/libxc/libxc/-/archive/7.0.0/libxc-7.0.0.tar.gz"
ARCHIVE_SHA256 = "8d4e343041c9cd869833822f57744872076ae709a613c118d70605539fb13a77"
SOURCE_ROOT = ROOT / libxc_bulk.SOURCE_ASSET
REPORT = ROOT / "docs/libxc_bulk_import.md"
_TYPE = re.compile(r"\(\*\s*type:\s*([^*]+?)\s*\*\)")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def extract_archive(archive: Path, destination: Path) -> None:
    """Select raw inputs from an exact archive; never extract arbitrary paths."""
    if digest(archive) != ARCHIVE_SHA256:
        raise ValueError("Libxc archive SHA-256 does not match the pinned 7.0.0 input")
    prefix = f"libxc-{REVISION}"
    with tarfile.open(archive, "r:gz") as tar:
        members: dict[str, tarfile.TarInfo] = {}
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            if not path.parts or path.parts[0] != prefix or ".." in path.parts:
                raise ValueError("unexpected or unsafe Libxc archive path")
            if len(path.parts) == 1:
                continue
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if relative in members:
                raise ValueError("duplicate archive member")
            members[relative] = member
        maple = sorted(
            name
            for name in members
            if name.startswith("maple/") and name.endswith(".mpl")
        )
        if not maple:
            raise ValueError("archive has no Maple inventory")
        keep = set(maple) | {
            "COPYING",
            "src/util.h",
            "src/util.c",
            "src/functionals.c",
            "src/xc.h",
        }
        keep.update(
            name
            for entry in maple
            if (name := f"src/{PurePosixPath(entry).stem}.c") in members
        )
        if any(name not in members for name in keep):
            raise ValueError(
                "Libxc archive lacks a required parameter/provenance input"
            )
        if any(
            not members[name].isfile() or members[name].size > 2_000_000
            for name in keep
        ):
            raise ValueError("non-regular or oversized Libxc source input")
        if sum(members[name].size for name in keep) > 16_000_000:
            raise ValueError("Libxc source projection exceeds the bounded size")
        if destination.exists():
            existing = {
                path.relative_to(destination).as_posix()
                for path in destination.rglob("*")
                if path.is_file()
            }
            if existing - keep:
                raise ValueError(
                    "source destination contains files outside the pinned projection"
                )
        for name in sorted(keep):
            target = destination / name
            if target.is_symlink():
                raise ValueError("source destination must not contain symlinks")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.resolve().relative_to(destination.resolve())
            stream = tar.extractfile(members[name])
            if stream is None:
                raise ValueError("could not read pinned archive input")
            with stream:
                target.write_bytes(stream.read())


def make_catalog(root: Path) -> dict[str, Any]:
    """Actually import/differentiate every eligible C registration in both spins."""
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    if not paths or not (root / "maple/util.mpl").is_file():
        raise ValueError("missing or empty pinned Libxc source tree")
    if any(path.is_symlink() for path in paths):
        raise ValueError("pinned source projection must contain regular files")
    files = {path.relative_to(root).as_posix(): digest(path) for path in paths}
    header = (root / "src/util.h").read_text(encoding="utf-8")
    registrations: list[dict[str, Any]] = []
    source_inventory: list[dict[str, Any]] = []
    for entry in sorted(root.glob("maple/**/*.mpl")):
        text = entry.read_text(encoding="utf-8")
        match = _TYPE.search(text)
        functional_type = match[1].strip() if match else None
        logical = entry.relative_to(root).as_posix()
        item: dict[str, Any] = {"entry": logical, "type": functional_type or "support"}
        owner = root / "src" / f"{entry.stem}.c"
        records = []
        if functional_type and owner.is_file():
            records = extract_registrations(
                owner.read_text(encoding="utf-8"), text, header
            )
        item["registrations"] = len(records)
        if functional_type and not records:
            item["reason"] = "no matching C registration; helper or unsupported owner"
        source_inventory.append(item)
        for record in records:
            record.update(entry=logical, owner=owner.relative_to(root).as_posix())
            record["graph_status"] = "blocked"
            if record["metadata_status"] == "bound":
                if functional_type != f"{record['family']}_exc":
                    record["reason"] = "source is not a matching energy functional"
                else:
                    try:
                        nodes = {}
                        for spin in ("polarized", "unpolarized"):
                            program = libxc_bulk.build_record(
                                record, files, root, spin=spin
                            )
                            roots = program.roots(1)
                            nodes[spin] = len(program.graph.topological_order(roots))
                        record.update(graph_status="imported", graph_nodes=nodes)
                    except (ValueError, ArithmeticError, RecursionError) as error:
                        record["reason"] = f"{type(error).__name__}: {error}".replace(
                            str(root), "<pinned>"
                        )
            registrations.append(record)
    registrations.sort(key=lambda record: record["name"])
    if not registrations or len({record["name"] for record in registrations}) != len(
        registrations
    ):
        raise ValueError("missing or duplicate Libxc registration inventory")
    imported = [
        record for record in registrations if record["graph_status"] == "imported"
    ]
    return {
        "schema": "vibeqc.libxc-bulk",
        "schema_version": 1,
        "bulk_semantics": libxc_bulk.BULK_SEMANTICS,
        "importer_semantics": libxc_maple.IMPORTER_SEMANTICS,
        "parameter_binding_semantics": C_BINDING_SEMANTICS,
        "upstream": {
            "revision": REVISION,
            "archive_url": ARCHIVE_URL,
            "archive_sha256": ARCHIVE_SHA256,
        },
        "counts": {
            "maple_files": len(source_inventory),
            "energy_source_files": sum(
                item["type"] in ("lda_exc", "gga_exc", "mgga_exc")
                for item in source_inventory
            ),
            "registrations": len(registrations),
            "graph_imported_registrations": len(imported),
            "graph_imported_source_files": len({item["entry"] for item in imported}),
            "blocked_registrations": len(registrations) - len(imported),
        },
        "source_files": files,
        "source_inventory": source_inventory,
        "registrations": registrations,
    }


def render_report(catalog: dict[str, Any]) -> str:
    counts = catalog["counts"]
    lines = [
        "# Bulk Libxc Maple import",
        "",
        "<!-- Generated by tools/import_libxc_bulk.py; do not edit inventory numbers. -->",
        "",
        (
            f"Pinned Libxc {REVISION}: **{counts['maple_files']} Maple files**, including "
            f"**{counts['energy_source_files']} energy-source files**. The C owners contain "
            f"**{counts['registrations']} registrations** (including parameter variants)."
        ),
        "",
        (
            f"**{counts['graph_imported_registrations']} registrations from "
            f"{counts['graph_imported_source_files']} distinct mathematical source files** "
            "actually lower and differentiate through the shared Graph in both spin layouts. "
            f"**{counts['blocked_registrations']} registrations remain explicitly blocked.**"
        ),
        "",
        (
            "These are real compiler imports, not static parser candidates or new public SCF methods. "
            "One source can define multiple parameterized registrations; those are not counted as "
            "distinct Maple equations. No runtime Libxc, Maple, PySCF, Torch or GPU is required."
        ),
        "",
        "## Use",
        "",
        "```python",
        "from vibeqc_compiler.xc.libxc_bulk import available_functionals, build_bulk_program",
        "program = build_bulk_program('GGA_X_PBE_SOL', spin='polarized')",
        "roots = program.roots(2)  # E, vxc, then upper-triangular feature Hessian",
        "c_source = program.emit_source(2)",
        "cuda_source = program.emit_source(2, cuda=True)",
        "```",
        "",
        (
            "Features are density only for LDA; density and sigma for GGA; density, sigma, "
            "laplacian and tau for meta-GGA. Polarized order is rho_a/rho_b, "
            "sigma_aa/sigma_ab/sigma_bb, lapl_a/lapl_b, tau_a/tau_b. "
            "Unpolarized order is rho, sigma, lapl, tau; absent family features are omitted. "
            "Energy is per volume and derivatives are with respect to these physical features."
        ),
        "",
        "## Domain and admission",
        "",
        (
            "The import contract is the ordinary positive-density/positive-gradient interior, "
            "not vacuum, fully polarized, zero-gradient or extreme-tail production continuation. "
            "Independent point fixtures do not constitute molecular SCF, forces, response, GPU "
            "execution, performance or complete-domain qualification. Existing FunctionalSpec/MethodIR "
            "and public Calculator admission are unchanged (#744/#745)."
        ),
        "",
        (
            "Only direct-copy homogeneous-double C parameter layouts are admitted. Custom setters, "
            "hybrid composition, non-3D and kinetic methods remain explicit blockers. "
            "External display names are not assumed to match the C struct layout. "
            "Unknown bindings and unsupported Maple constructs never fall back to handwritten equations."
        ),
        "",
        "## Reproduce",
        "",
        "```bash",
        "# One-time, explicit acquisition outside normal builds:",
        f"curl -L --fail -o libxc-7.0.0.tar.gz {ARCHIVE_URL}",
        "python tools/import_libxc_bulk.py --archive libxc-7.0.0.tar.gz",
        "# Subsequent regeneration and verification are offline:",
        "python tools/import_libxc_bulk.py --check",
        "python tools/source_registry.py verify",
        "```",
        "",
        (
            f"Archive SHA-256: `{ARCHIVE_SHA256}`. The compact raw-source projection retains "
            "the complete Maple inventory, C owners, utility/header semantics and license; "
            "it excludes generated Libxc C kernels and is not compiled into the VibeQC runtime."
        ),
        "",
        "## Remaining blocker groups",
        "",
        "| Reason | Registrations |",
        "| --- | ---: |",
    ]
    reasons = Counter(
        record["reason"]
        for record in catalog["registrations"]
        if record["graph_status"] == "blocked"
    )
    lines.extend(
        f"| {reason.replace('|', '&#124;').replace(chr(10), ' ')} | {count} |"
        for reason, count in sorted(reasons.items())
    )
    lines.extend(
        [
            "",
            (
                "The complete per-registration source, default bindings, provenance and "
                "status are in `python/vibeqc_compiler/xc/libxc_bulk_catalog.json`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def register_product(catalog: dict[str, Any]) -> None:
    """Use the existing common scientific source/product freshness contract."""
    registry = source_registry._load()
    source_id = "libxc-7.0.0-fulltree"
    registry["sources"][source_id] = {
        "kind": "file-set",
        "license": "MPL-2.0",
        "revision": REVISION,
        "repository": "https://gitlab.com/libxc/libxc",
        "local_root": libxc_bulk.SOURCE_ASSET,
        "files": {
            name: {
                "upstream_path": name,
                "sha256": sha,
                "url": f"https://gitlab.com/libxc/libxc/-/raw/{REVISION}/{name}",
            }
            for name, sha in catalog["source_files"].items()
        },
    }
    generator = Path(__file__).relative_to(ROOT).as_posix()
    registry["products"]["libxc-bulk-graph-catalog"] = {
        "inputs": [source_id],
        "input_identity_sha256": source_registry._product_input_identity(
            registry["sources"], [source_id]
        ),
        "generator": generator,
        "generator_sha256": digest(Path(__file__)),
        "canonical_inputs": {
            path: digest(ROOT / path)
            for path in (
                "tools/libxc_bulk_metadata.py",
                "python/vibeqc_compiler/xc/libxc_bulk.py",
                "python/vibeqc_compiler/xc/libxc_maple.py",
            )
        },
        "outputs": {
            path.relative_to(ROOT).as_posix(): digest(path)
            for path in (libxc_bulk.CATALOG_PATH, REPORT)
        },
    }
    source_registry.REGISTRY.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.archive and args.check:
        parser.error("--archive and --check are mutually exclusive")
    if args.archive:
        extract_archive(args.archive, SOURCE_ROOT)
    catalog = make_catalog(SOURCE_ROOT)
    outputs = {
        libxc_bulk.CATALOG_PATH: json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        REPORT: render_report(catalog),
    }
    if args.check:
        for path, expected in outputs.items():
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                raise SystemExit(f"stale bulk import product: {path.relative_to(ROOT)}")
    else:
        for path, content in outputs.items():
            path.write_text(content, encoding="utf-8")
        register_product(catalog)
    print(json.dumps(catalog["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
