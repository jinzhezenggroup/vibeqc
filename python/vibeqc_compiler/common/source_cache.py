"""Atomic publication of immutable, content-addressed generated sources."""

import os
import tempfile
import typing
from pathlib import Path


def cache_source(path: typing.Any, source: typing.Any) -> None:
    """Publish complete bytes; reject a corrupt existing cache entry."""
    if path.exists():
        if path.read_text() != source:
            raise ValueError("native source identity mismatch")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".source-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
