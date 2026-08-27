"""Resolve generated-shell CUDA profiles without unsafe cross-SM reuse."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

PORTABLE_PROFILE = "portable_cuda"
_ARCHITECTURE_PATTERN = re.compile(
    r"^(?:sm_)?(?P<digits>[0-9]+)(?:-(?:real|virtual))?$"
)


@dataclass(frozen=True, slots=True)
class ProfileResolution:
    """One requested CUDA target and the production profile selected for it."""

    requested_architecture: str
    selected_profile: str
    tuned: bool
    reason: str


def normalize_cuda_architecture(value: str) -> str:
    """Normalize CMake/NVCC architecture spelling to a manifest key when possible."""

    architecture = value.strip()
    if not architecture:
        raise ValueError("CUDA architecture must not be empty")
    match = _ARCHITECTURE_PATTERN.fullmatch(architecture)
    if match is not None:
        return f"sm_{match.group('digits')}"
    return architecture


def _load_architectures(path: Path) -> dict[str, object]:
    """Load the schema-v2 architecture map used by production code generation."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("CUDA profile resolution requires a schema-v2 manifest")
    architectures = payload.get("architectures")
    if not isinstance(architectures, dict):
        raise ValueError("production manifest requires an architectures object")
    return architectures


def _is_tuned_profile(profile: object) -> bool:
    """Return whether a profile contains at least one measured production kernel."""

    if not isinstance(profile, dict):
        return False
    kernels = profile.get("kernels")
    return isinstance(kernels, list) and bool(kernels)


def resolve_profile(
    manifest: Path,
    architecture: str,
    requested_profile: str = "auto",
    *,
    require_tuned: bool = False,
) -> ProfileResolution:
    """Resolve an exact measured profile or the zero-AOT portable fallback.

    Exact architecture matches are the only tuned profiles accepted by this
    first-stage resolver. It never reuses an ``sm_120`` schedule for another
    compute capability. Multi-profile fat binaries and family-compatible
    profiles are intentionally left to the follow-up runtime-dispatch work.
    """

    normalized_architecture = normalize_cuda_architecture(architecture)
    profiles = _load_architectures(manifest)
    selected_request = requested_profile.strip()
    if not selected_request:
        raise ValueError("requested CUDA profile must not be empty")

    if selected_request == "auto":
        exact = profiles.get(normalized_architecture)
        if _is_tuned_profile(exact):
            resolution = ProfileResolution(
                requested_architecture=normalized_architecture,
                selected_profile=normalized_architecture,
                tuned=True,
                reason="exact measured architecture profile",
            )
        else:
            resolution = ProfileResolution(
                requested_architecture=normalized_architecture,
                selected_profile=PORTABLE_PROFILE,
                tuned=False,
                reason="no exact measured profile; generated AOT kernels disabled",
            )
    elif selected_request == PORTABLE_PROFILE:
        resolution = ProfileResolution(
            requested_architecture=normalized_architecture,
            selected_profile=PORTABLE_PROFILE,
            tuned=False,
            reason="portable profile explicitly requested",
        )
    else:
        selected_profile = normalize_cuda_architecture(selected_request)
        if selected_profile != normalized_architecture:
            raise ValueError(
                "refusing to use CUDA profile "
                f"{selected_profile!r} for target {normalized_architecture!r}"
            )
        if not _is_tuned_profile(profiles.get(selected_profile)):
            raise ValueError(
                f"production manifest has no tuned profile for {selected_profile}"
            )
        resolution = ProfileResolution(
            requested_architecture=normalized_architecture,
            selected_profile=selected_profile,
            tuned=True,
            reason="exact profile explicitly requested",
        )

    if require_tuned and not resolution.tuned:
        raise ValueError(
            f"no tuned generated-shell profile matches "
            f"{resolution.requested_architecture}"
        )
    return resolution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--require-tuned", action="store_true")
    arguments = parser.parse_args()
    try:
        resolution = resolve_profile(
            arguments.manifest,
            arguments.architecture,
            arguments.profile,
            require_tuned=arguments.require_tuned,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(
        f"{resolution.selected_profile};"
        f"{resolution.requested_architecture};"
        f"{int(resolution.tuned)}"
    )


if __name__ == "__main__":
    main()
