"""Read scientific JSON whose large lists are stored in named companion files.

Lists are grouped by workload or observable, with their original ordering and
values preserved. Parts are ordinary reviewable JSON, not byte fragments.
Storage metadata does not change the reconstructed validation schema.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from .retention import digest, safe_relative


def decode_json(data: bytes, *, path: str | Path | None = None) -> Any:
    """Decode ordinary or gzip-compressed JSON."""
    name = str(path) if path is not None else ""
    if name.endswith(".gz") or data.startswith(b"\x1f\x8b"):
        data = gzip.decompress(data)
    return json.loads(data)


def load_json(path: Path) -> Any:
    """Load ordinary or gzip-compressed JSON from the checkout."""
    path = Path(path)
    return decode_json(path.read_bytes(), path=path)


def decode_record(
    data: bytes, files: dict[str, bytes], *, path: str | Path | None = None
) -> dict:
    """Restore list fields from checksum-pinned JSON parts in a publication.

    Part paths are relative to the record's directory. A field must be absent
    from the main record so storage metadata cannot overwrite scientific data.
    Parts cannot recursively include files or redefine other fields.
    """
    record = decode_json(data, path=path)
    parts = record.pop("record_parts", {})
    for field, entries in parts.items():
        if field in record or not entries:
            raise ValueError("record parts overwrite a field or contain no files")
        values = []
        seen = set()
        for entry in entries:
            path = safe_relative(entry["path"])
            if path in seen:
                raise ValueError("duplicate record part")
            seen.add(path)
            raw = files[path]
            if len(raw) != entry["bytes"] or digest(raw) != entry["sha256"]:
                raise ValueError(f"record part checksum/size mismatch: {path}")
            part = decode_json(raw, path=path)
            if not isinstance(part, list):
                raise TypeError("record part must contain a list")
            values.extend(part)
        record[field] = values
    return record


def load_record(path: Path) -> dict:
    """Load a plain or partitioned scientific record from the local checkout."""
    path = Path(path)
    data = path.read_bytes()
    parts = decode_json(data, path=path).get("record_parts", {})
    files = {}
    for entries in parts.values():
        for entry in entries:
            name = safe_relative(entry["path"])
            target = path.parent / name
            if not target.resolve().is_relative_to(path.parent.resolve()):
                raise ValueError("record part escapes the publication directory")
            files[name] = target.read_bytes()
    return decode_record(data, files, path=path)
