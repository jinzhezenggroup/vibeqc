"""Deterministic cache identity for optional NVRTC long-tail kernels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class NvrtcCacheSpec:
    """Every input that can change generated PTX or device compatibility."""

    generator_abi: str
    shell_class: tuple[int, int, int, int]
    derivative_centers: tuple[int, ...]
    precision_policy: str
    screening_policy: str
    source_digest: str
    compute_capability: str
    nvrtc_version: str
    driver_version: str


def nvrtc_cache_key(specification: NvrtcCacheSpec) -> str:
    """Return a stable content-addressed key for one compiled kernel."""

    payload = json.dumps(
        asdict(specification), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
