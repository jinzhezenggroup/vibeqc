"""Provenance headers cannot authenticate modified binary table contents."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest
from vibeqc_compiler.common import d3_data


def synthetic() -> bytes:
    counts = (86, 3741, 237, 28455, 86, 3741, 86)
    header = struct.pack(
        "<8s40s32s32s7I", b"VQD3BIN1", b"a" * 40, b"b" * 32, b"c" * 32, *counts
    )
    return (
        header
        + struct.pack("<IB", 0, 1) * 86
        + struct.pack("<IBB", 0, 1, 1) * 3741
        + struct.pack("<d", 1.25) * sum(counts[2:])
    )


@pytest.mark.parametrize(
    "offset", [8, 48, 80, 140, 145, 570, 577, 23016, 24912, 25200, 253240, 283848]
)
def test_changed_body_cannot_keep_trusted_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offset: int
) -> None:
    original = synthetic()
    # A synthetic trusted product isolates the reader contract from chemistry.
    monkeypatch.setattr(
        d3_data, "D3_PRODUCTION_SHA256", hashlib.sha256(original).hexdigest(), raising=False
    )
    changed = bytearray(original)
    changed[offset] ^= 1
    path = tmp_path / "product.bin"
    path.write_bytes(changed)
    with pytest.raises(ValueError, match="digest"):
        d3_data.load_d3_production_data(path)


def test_trusted_binary_is_decoded_without_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = synthetic()
    monkeypatch.setattr(
        d3_data, "D3_PRODUCTION_SHA256", hashlib.sha256(raw).hexdigest(), raising=False
    )
    path = tmp_path / "product.bin"
    path.write_bytes(raw)
    data = d3_data.load_d3_production_data(path)
    assert data.source_revision == "a" * 40
    assert data.table_sha256 == (b"b" * 32).hex()
    assert len(data.c6) == 28455 and set(data.c6) == {1.25}
    assert data.covalent_radii == (1.25,) * 86
    assert path.read_bytes() == raw
