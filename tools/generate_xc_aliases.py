#!/usr/bin/env python3
"""Generate user-facing DFT aliases from pinned PySCF/Libxc metadata."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import tomllib

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from vibeqc_compiler.method.spec import METHOD_CATALOG
from vibeqc_compiler.xc.spec import VERSION as XC_VERSION

OUTPUT = ROOT / "python/vibeqc_compiler/method/_generated_xc_aliases.py"
FAMILY_PREFIXES = (
    "HYB_MGGA_XC_",
    "HYB_GGA_XC_",
    "HYB_LDA_XC_",
    "MGGA_XC_",
    "GGA_XC_",
    "LDA_XC_",
)


def _pinned_pyscf_version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = data["project"]["optional-dependencies"]["reference-test"]
    pins = [dep.removeprefix("pyscf==") for dep in deps if dep.startswith("pyscf==")]
    if len(pins) != 1:
        raise RuntimeError("reference-test must pin exactly one PySCF version")
    return pins[0]


def _signature(libxc: Any, name: str) -> tuple[Any, ...] | None:
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            hybrid, functionals = libxc.parse_xc(name)
    except (KeyError, ValueError):
        return None
    return (
        tuple(float(value) for value in hybrid),
        tuple(
            (int(identifier), float(coefficient))
            for identifier, coefficient in functionals
        ),
    )


def _candidate_names(libxc: Any) -> set[str]:
    candidates = set(libxc.XC_ALIAS)
    candidates.update(name for name in libxc.XC_CODES if "_" not in name)
    for name in libxc.XC_CODES:
        for prefix in FAMILY_PREFIXES:
            if name.startswith(prefix):
                candidates.add(name[len(prefix) :])
                break
    return candidates


def _build_aliases(libxc: Any) -> dict[str, str]:
    by_signature: dict[tuple[Any, ...], str] = {}
    for canonical, spec in METHOD_CATALOG.items():
        if any((spec.dispersion, spec.nonlocal_correlation, spec.basis, spec.gcp)):
            continue
        signature = _signature(libxc, canonical)
        if signature is None:
            continue
        previous = by_signature.get(signature)
        if previous is not None and previous != canonical:
            raise RuntimeError(
                f"canonical methods {previous!r} and {canonical!r} have the same "
                "PySCF/Libxc signature"
            )
        by_signature[signature] = canonical

    aliases: dict[str, str] = {}
    for candidate in sorted(_candidate_names(libxc)):
        signature = _signature(libxc, candidate)
        canonical = by_signature.get(signature) if signature is not None else None
        if canonical is not None and candidate != canonical:
            aliases[candidate] = canonical
    return aliases


def _render() -> str:
    import pyscf
    from pyscf.dft import libxc

    expected_pyscf = _pinned_pyscf_version()
    if pyscf.__version__ != expected_pyscf:
        raise RuntimeError(
            f"expected PySCF {expected_pyscf}, found {pyscf.__version__}"
        )
    expected_libxc = XC_VERSION.split("/", 1)[0].removeprefix("libxc-")
    if libxc.__version__ != expected_libxc:
        raise RuntimeError(
            f"expected Libxc {expected_libxc}, found {libxc.__version__}"
        )

    module_path = Path(libxc.__file__).resolve()
    module_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    aliases = _build_aliases(libxc)

    lines = [
        '"""Generated XC method aliases from pinned PySCF/Libxc metadata.',
        "",
        "Do not edit by hand. Regenerate with tools/generate_xc_aliases.py.",
        '"""',
        "",
        "from types import MappingProxyType",
        "",
        "UPSTREAM_XC_ALIAS_PROVENANCE = MappingProxyType(",
        "    {",
        f'        "pyscf_version": {json.dumps(pyscf.__version__)},',
        f'        "libxc_version": {json.dumps(libxc.__version__)},',
        '        "module": "pyscf.dft.libxc",',
        f'        "module_sha256": {json.dumps(module_sha256)},',
        "    }",
        ")",
        "",
        "METHOD_ALIASES = MappingProxyType(",
        "    {",
    ]
    lines.extend(
        f"        {json.dumps(alias)}: {json.dumps(canonical)},"
        for alias, canonical in aliases.items()
    )
    lines.extend(["    }", ")", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked-in generated alias table is stale",
    )
    args = parser.parse_args()
    rendered = _render()
    if args.check:
        if OUTPUT.read_text() != rendered:
            print(f"{OUTPUT.relative_to(ROOT)} is stale", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
