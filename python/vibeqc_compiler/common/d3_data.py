"""Deterministic compact D3 production-table container."""

from __future__ import annotations

import hashlib
import struct
import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from pathlib import Path

# Pinned output of upstream/manifest.json's d3-production-data product.
# Header source digests are metadata; authenticate the complete stored bytes.
D3_PRODUCTION_SHA256 = (
    "19aed2b11b39a29f6e1c9a8cb442cb78ef3772488edc0c003e7640a5b198e945"
)
MAGIC = b"VQD3BIN1"
_HEADER = struct.Struct("<8s40s32s32s7I")
_ELEMENT = struct.Struct("<IB")
_PAIR = struct.Struct("<IBB")


@dataclass(frozen=True)
class D3ElementData:
    reference_offset: int
    reference_count: int


@dataclass(frozen=True)
class D3PairData:
    c6_offset: int
    first_reference_count: int
    second_reference_count: int


@dataclass(frozen=True)
class D3ProductionData:
    source_revision: str
    table_sha256: str
    radii_sha256: str
    elements: tuple[D3ElementData, ...]
    pairs: tuple[D3PairData, ...]
    coordination_numbers: tuple[float, ...]
    c6: tuple[float, ...]
    r4r2: tuple[float, ...]
    vdw_radii: tuple[float, ...]
    covalent_radii: tuple[float, ...]


def _doubles(raw: bytes, offset: int, count: int) -> tuple[tuple[float, ...], int]:
    end = offset + 8 * count
    if end > len(raw):
        raise ValueError("truncated D3 production data")
    return tuple(struct.unpack_from(f"<{count}d", raw, offset)), end


def load_d3_production_data(path: Path) -> D3ProductionData:
    raw = path.read_bytes()
    if len(raw) < _HEADER.size:
        raise ValueError("truncated D3 production-data header")
    magic, revision, table, radii, ne, npair, nref, nc6, nr4, nvdw, nrad = (
        _HEADER.unpack_from(raw)
    )
    if magic != MAGIC:
        raise ValueError("unsupported D3 production-data format")
    if (ne, npair, nref, nc6, nr4, nvdw, nrad) != (86, 3741, 237, 28455, 86, 3741, 86):
        raise ValueError("unexpected D3 production-data shape")
    off = _HEADER.size
    elements = []
    pairs = []
    for _ in range(ne):
        if off + _ELEMENT.size > len(raw):
            raise ValueError("truncated D3 element table")
        a, b = _ELEMENT.unpack_from(raw, off)
        off += _ELEMENT.size
        elements.append(D3ElementData(a, b))
    for _ in range(npair):
        if off + _PAIR.size > len(raw):
            raise ValueError("truncated D3 pair table")
        a, b, c = _PAIR.unpack_from(raw, off)
        off += _PAIR.size
        pairs.append(D3PairData(a, b, c))
    cn, off = _doubles(raw, off, nref)
    c6, off = _doubles(raw, off, nc6)
    r4, off = _doubles(raw, off, nr4)
    vdw, off = _doubles(raw, off, nvdw)
    cr, off = _doubles(raw, off, nrad)
    if off != len(raw):
        raise ValueError("D3 production data has trailing bytes")
    if hashlib.sha256(raw).hexdigest() != D3_PRODUCTION_SHA256:
        raise ValueError("D3 production data digest mismatch")
    return D3ProductionData(
        revision.decode("ascii"),
        table.hex(),
        radii.hex(),
        tuple(elements),
        tuple(pairs),
        cn,
        c6,
        r4,
        vdw,
        cr,
    )
