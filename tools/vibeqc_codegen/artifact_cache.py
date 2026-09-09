"""Bounded, atomic, explicitly local executable cache for optional backends.

CUDA official/user-local dispatch and NVRTC keys remain unchanged. The cache
uses the established profile atomic publication helper, but never auto-loads a
remote profile or activates a backend for a scientific calculation.
"""

import base64
import hashlib
import json
import os
import stat
from dataclasses import asdict
from pathlib import Path

from .runtime_backend import CompiledArtifactIdentity

_MAXIMUM_BINARY_BYTES = 64 << 20
_MAXIMUM_RECORD_BYTES = 90 << 20


class LocalArtifactCache:
    """A private local cache selected explicitly by the caller, with full identities."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = self.directory.stat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o022
        ):
            raise ValueError(
                "executable cache must be an owned directory without shared write access"
            )

    def install(self, identity: CompiledArtifactIdentity, binary: bytes):
        """Publish one complete checksummed executable record with fsync and replace."""
        from vibeqc.profiles import atomic_json

        if (
            not isinstance(binary, bytes)
            or not 0 < len(binary) <= _MAXIMUM_BINARY_BYTES
        ):
            raise ValueError("bounded nonempty binary bytes required")
        payload = {
            "schema": "vibeqc.backend_artifact",
            "version": 1,
            "identity": asdict(identity),
            "sha256": hashlib.sha256(binary).hexdigest(),
            "bytes": len(binary),
            "binary": base64.b64encode(binary).decode("ascii"),
        }
        path = self.directory / (identity.key + ".json")
        atomic_json(path, payload)
        return path

    def load(self, expected: CompiledArtifactIdentity):
        """Verify ownership, bounds, compatibility and checksum before returning bytes."""
        path = self.directory / (expected.key + ".json")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o022
            ):
                raise ValueError("cache record is not a trusted owned regular file")
            if info.st_size > _MAXIMUM_RECORD_BYTES:
                raise ValueError("cache record exceeds its read limit")
            raw = stream.read(_MAXIMUM_RECORD_BYTES + 1)
        if len(raw) > _MAXIMUM_RECORD_BYTES:
            raise ValueError("cache record exceeded its read limit")
        payload = json.loads(raw)
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "vibeqc.backend_artifact"
            or type(payload.get("version")) is not int
            or payload.get("version") != 1
        ):
            raise ValueError("unsupported executable cache record")
        try:
            actual = CompiledArtifactIdentity(**payload["identity"])
            if (
                not isinstance(payload["binary"], str)
                or type(payload["bytes"]) is not int
                or not isinstance(payload["sha256"], str)
            ):
                raise ValueError("invalid executable cache metadata types")
            binary = base64.b64decode(payload["binary"], validate=True)
        except (KeyError, TypeError) as error:
            raise ValueError("malformed executable cache metadata") from error
        expected.require_compatible(actual)
        if (
            not 0 < len(binary) <= _MAXIMUM_BINARY_BYTES
            or len(binary) != payload["bytes"]
            or hashlib.sha256(binary).hexdigest() != payload["sha256"]
        ):
            raise ValueError("cached executable is truncated or corrupt")
        return binary
