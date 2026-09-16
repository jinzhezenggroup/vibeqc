"""Compile-time experimental raw-value mapping; promotion requires endpoint evidence."""

import json
from pathlib import Path

from ..df_value_candidates import VALUE_CLASSES

VALUE_MANIFEST = Path(__file__).resolve().parents[1] / "production_df_values.json"


def load_value_manifest(path=VALUE_MANIFEST):
    """Validate an explicit bounded architecture/class policy without device probing."""
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != 1 or payload.get("architecture") != "sm_120":
        raise ValueError("unsupported value production manifest")
    if type(payload.get("qualified")) is not bool or payload.get("raw_lanes") not in (
        1,
        4,
        32,
    ):
        raise ValueError("invalid value production qualification/schedule")
    keys = {"".join(map(str, a)) for a in VALUE_CLASSES}
    if set(payload.get("kernels", {})) != keys or any(
        v not in ("generic", "polynomial", "rys") for v in payload["kernels"].values()
    ):
        raise ValueError("complete supported value class mapping required")
    required = ["candidate_report"]
    if payload["qualified"]:
        required += [
            "generator_sha256",
            "toolchain",
            "endpoint_768",
            "source_endpoint",
            "sanitizer",
            "fixtures",
        ]
    evidence = payload.get("provenance", {})
    if any(not isinstance(evidence.get(k), str) or not evidence[k] for k in required):
        raise ValueError("complete value qualification evidence required")
    return payload
