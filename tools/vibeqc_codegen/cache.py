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


def integral_cache_key(integral) -> str:
    """Hash versioned scientific intent through the existing content-addressing scheme.

    This identity includes physical bindings, external charges, tensor strides,
    weight source/signs, and budgets. It describes intent, not compiled support.
    The legacy NVRTC key and production ABI remain unchanged because their
    symbols, layouts, and generated sources are byte-identical.
    """
    from .ir_serialization import integral_to_payload

    payload = json.dumps(
        integral_to_payload(integral),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
