"""Compile-time experimental raw-value mappings keyed by CUDA architecture."""

from __future__ import annotations

import json
import re
import typing
from pathlib import Path

from ...common.cuda_target import normalize_cuda_architecture
from ..df_value_candidates import VALUE_CLASSES

VALUE_MANIFEST = Path(__file__).resolve().parents[1] / "production_df_values.json"


def _validate_profile(profile: typing.Any) -> None:
    """Validate one measured/candidate architecture profile."""
    if not isinstance(profile, dict) or type(profile.get("qualified")) is not bool:
        raise ValueError("explicit value production qualification status required")
    if type(profile.get("raw_lanes")) is not int or profile["raw_lanes"] not in (
        1,
        4,
        32,
    ):
        raise ValueError("invalid value production schedule")
    keys = {"".join(map(str, angular)) for angular in VALUE_CLASSES}
    kernels = profile.get("kernels", {})
    if set(kernels) != keys or any(
        value not in ("generic", "polynomial", "rys") for value in kernels.values()
    ):
        raise ValueError("complete supported value class mapping required")
    required = ["candidate_report"]
    if profile["qualified"]:
        required += [
            "generator_sha256",
            "toolchain",
            "endpoint_768",
            "source_endpoint",
            "sanitizer",
            "fixtures",
        ]
    evidence = profile.get("provenance", {})
    if any(
        not isinstance(evidence.get(key), str) or not evidence[key] for key in required
    ):
        raise ValueError("complete value qualification evidence required")


def load_value_manifest(path: typing.Any = VALUE_MANIFEST) -> typing.Any:
    """Validate architecture-qualified value policies without device probing."""
    payload = json.loads(Path(path).read_text())
    architectures = payload.get("architectures")
    if payload.get("schema_version") != 2 or not isinstance(architectures, dict):
        raise ValueError("unsupported value production manifest")
    for architecture, profile in architectures.items():
        if not re.fullmatch(r"sm_[1-9][0-9]+", architecture):
            raise ValueError("invalid target architecture")
        _validate_profile(profile)
    return payload


def generic_value_profile() -> dict[str, typing.Any]:
    """Return the safe profile for an architecture without measured value tuning."""
    keys = ("".join(map(str, angular)) for angular in VALUE_CLASSES)
    return {
        "qualified": False,
        "raw_lanes": 1,
        "kernels": {key: "generic" for key in keys},
        "provenance": {},
    }


def resolve_value_profile(payload: typing.Any, architecture: str | int) -> typing.Any:
    """Resolve an exact architecture profile or the generic portable fallback."""
    canonical = normalize_cuda_architecture(architecture)
    profile = payload["architectures"].get(canonical)
    return profile if profile is not None else generic_value_profile()
