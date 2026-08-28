"""Extract one fixed-root Rys interpolation slice from GPU4PySCF.

GPU4PySCF stores roots one through ten in shared CUDA tables.  Production
VibeQC kernels only need the exact root count implied by one shell class, so
embedding the complete table would increase generated CUDA size and compile
time unnecessarily.  This helper makes the checked-in fixed-root modules
reproducible while preserving the upstream Apache attribution in each output.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

DEGREE = 13
INTERVALS = 40
GPU4PYSCF_COMMIT = "db6bb7f12bf4b6465b7c5ff0a9741bc2be08318d"
GPU4PYSCF_SOURCE_SHA256 = (
    "42d064e05fd7e8bf233ef295556b6ce10162053a74e778c0c39b6ac7213725d0"
)
_ARRAY_NAMES = (
    "ROOT_SMALLX_R0",
    "ROOT_SMALLX_R1",
    "ROOT_SMALLX_W0",
    "ROOT_SMALLX_W1",
    "ROOT_LARGEX_R_DATA",
    "ROOT_LARGEX_W_DATA",
    "ROOT_RW_DATA",
)
_NUMBER = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def _read_cuda_array(source: str, name: str) -> tuple[float, ...]:
    """Return one numeric CUDA initializer while ignoring line comments."""

    match = re.search(
        rf"\b{name}\s*\[\s*\]\s*=\s*\{{(?P<body>.*?)\}};",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(f"CUDA array {name!r} is missing")
    body = re.sub(r"//[^\n]*", "", match.group("body"))
    return tuple(float(token) for token in _NUMBER.findall(body))


def extract_fixed_root_tables(
    source: str, nroots: int
) -> dict[str, tuple[float, ...]]:
    """Extract the compact table slice addressed by GPU4PySCF's evaluator."""

    if not 1 <= nroots <= 10:
        raise ValueError("GPU4PySCF fixed-root tables support one through ten roots")
    arrays = {name: _read_cuda_array(source, name) for name in _ARRAY_NAMES}
    triangular_offset = nroots * (nroots - 1) // 2
    interpolation_width = (DEGREE + 1) * INTERVALS
    interpolation_offset = interpolation_width * nroots * (nroots - 1)
    interpolation_count = 2 * nroots * interpolation_width
    result = {
        name: values[triangular_offset : triangular_offset + nroots]
        for name, values in arrays.items()
        if name != "ROOT_RW_DATA"
    }
    result["ROOT_RW_DATA"] = arrays["ROOT_RW_DATA"][
        interpolation_offset : interpolation_offset + interpolation_count
    ]
    expected = {
        name: interpolation_count if name == "ROOT_RW_DATA" else nroots
        for name in _ARRAY_NAMES
    }
    for name, values in result.items():
        if len(values) != expected[name]:
            raise ValueError(
                f"CUDA array {name!r} ended before the nroots={nroots} slice"
            )
    return result


def _format_values(values: tuple[float, ...], columns: int = 4) -> str:
    """Format deterministic Python tuple contents."""

    return "\n".join(
        "    "
        + ", ".join(f"{value:.17e}" for value in values[begin : begin + columns])
        + ","
        for begin in range(0, len(values), columns)
    )


def emit_python_module(nroots: int, tables: dict[str, tuple[float, ...]]) -> str:
    """Emit one attributed, importable fixed-root data module."""

    prefix = f"RYS{nroots}"
    labels = {
        "ROOT_SMALLX_R0": "SMALLX_R0",
        "ROOT_SMALLX_R1": "SMALLX_R1",
        "ROOT_SMALLX_W0": "SMALLX_W0",
        "ROOT_SMALLX_W1": "SMALLX_W1",
        "ROOT_LARGEX_R_DATA": "LARGEX_R_DATA",
        "ROOT_LARGEX_W_DATA": "LARGEX_W_DATA",
        "ROOT_RW_DATA": "RW_DATA",
    }
    sections = [
        (
            f'"""{nroots}-root Rys interpolation data derived from GPU4PySCF.\n\n'
            "Copyright 2021-2024 The PySCF Developers. All Rights Reserved.\n"
            "Licensed under the Apache License, Version 2.0.  The numerical "
            "table is\n"
            f"the ``nroots == {nroots}`` slice of GPU4PySCF's "
            "``gvhf-rys/rys_roots_dat.cu``.\n"
            f"Source revision: ``{GPU4PYSCF_COMMIT}``.\n"
            f"Source SHA-256: ``{GPU4PYSCF_SOURCE_SHA256}``.\n"
            '"""\n'
        ),
        f"{prefix}_DEGREE = {DEGREE}\n{prefix}_INTERVALS = {INTERVALS}\n",
    ]
    for source_name in _ARRAY_NAMES:
        sections.append(
            f"{prefix}_{labels[source_name]} = (\n"
            f"{_format_values(tables[source_name])}\n)\n"
        )
    return "\n".join(sections)


def main() -> None:
    """Write one fixed-root Python module from an upstream CUDA table."""

    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--nroots", type=int, required=True)
    arguments = parser.parse_args()
    source_bytes = arguments.source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != GPU4PYSCF_SOURCE_SHA256:
        raise ValueError(
            "GPU4PySCF Rys source does not match the repository's locked "
            f"revision {GPU4PYSCF_COMMIT}: expected SHA-256 "
            f"{GPU4PYSCF_SOURCE_SHA256}, found {source_hash}"
        )
    source = source_bytes.decode("utf-8")
    module = emit_python_module(
        arguments.nroots,
        extract_fixed_root_tables(source, arguments.nroots),
    )
    arguments.output.write_text(module, encoding="utf-8")


if __name__ == "__main__":
    main()
