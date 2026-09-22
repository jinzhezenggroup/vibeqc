"""Write deterministic production CUDA bundles and registry artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibeqc_compiler.common.cuda_target import normalize_cuda_architecture

from .production_cost import _partition_production_selections
from .production_emission import emit_production_shard, emit_profile_shard
from .production_profile import (
    _profile_identifier,
    load_production_kernel_selections,
    resolve_production_profile,
)
from .production_registry import (
    emit_multi_registry_header,
    emit_multi_registry_source,
    emit_registry_header,
    emit_registry_source,
)
from .shell_spec import FUSED_SHELL_SPEC_BY_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


def write_production_bundles(
    manifest: Path,
    output_directory: Path,
    shard_count: int,
    architectures: Sequence[str],
    profile_by_architecture: Mapping[str, str] | None = None,
    unit_mode: str = "stable-shards",
    all_class_units: bool = False,
) -> tuple[Path, ...]:
    """Write independent, namespaced AOT bundles and one runtime registry.

    ``stable-shards`` is the release-compatible layout.  ``class`` is a
    development layout that emits one translation unit per shell class, which
    is useful when iterating on a heavy generated class without compiling its
    neighbors.  Both layouts use the same symbols and registry ABI.
    """

    if isinstance(shard_count, bool) or not isinstance(shard_count, int):
        raise TypeError("shard_count must be an integer")
    if shard_count < 1:
        raise ValueError("production shard count must be positive")
    if unit_mode not in ("stable-shards", "class"):
        raise ValueError("unit_mode must be 'stable-shards' or 'class'")
    normalized = tuple(
        sorted(
            {normalize_cuda_architecture(item) for item in architectures},
            key=lambda item: int(item.removeprefix("sm_")),
        )
    )
    if not normalized:
        raise ValueError("at least one CUDA architecture is required")
    requested = {
        normalize_cuda_architecture(key): value
        for key, value in (profile_by_architecture or {}).items()
    }
    profiles = tuple(
        resolve_production_profile(
            manifest,
            architecture,
            requested.get(architecture, "auto"),
        )
        for architecture in normalized
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for profile in profiles:
        identifier = _profile_identifier(profile.target.architecture)
        profile_directory = output_directory / profile.target.architecture
        profile_directory.mkdir(parents=True, exist_ok=True)
        if unit_mode == "class":
            selected_by_name = {
                selection.spec.name: selection for selection in profile.selections
            }
            names = (
                tuple(sorted(FUSED_SHELL_SPEC_BY_NAME))
                if all_class_units
                else tuple(sorted(selected_by_name))
            )
            units = tuple(
                (name, ((selected_by_name[name],) if name in selected_by_name else ()))
                for name in names
            )
            for name, unit in units:
                path = profile_directory / (
                    f"vibeqc_generated_shell_{identifier}_{name}.cu"
                )
                _write_if_changed(path, emit_profile_shard(profile, unit))
                outputs.append(path)
        else:
            shards = _partition_production_selections(profile.selections, shard_count)
            for index, shard in enumerate(shards):
                path = profile_directory / (
                    f"vibeqc_generated_shell_{identifier}_shard_{index}.cu"
                )
                _write_if_changed(path, emit_profile_shard(profile, shard))
                outputs.append(path)
    header = output_directory / "vibeqc_generated_shell_registry.hpp"
    source = output_directory / "vibeqc_generated_shell_registry.cu"
    _write_if_changed(header, emit_multi_registry_header(profiles))
    _write_if_changed(source, emit_multi_registry_source(profiles))
    outputs.extend((header, source))
    return tuple(outputs)


def _write_if_changed(path: Path, content: str) -> None:
    """Preserve timestamps when deterministic regeneration is byte-identical."""

    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def write_production_bundle(
    manifest: Path,
    output_directory: Path,
    shard_count: int,
    architecture: str | None = None,
    profile: str = "auto",
    unit_mode: str = "stable-shards",
    all_class_units: bool = False,
) -> tuple[Path, ...]:
    """Write deterministic build artifacts and return every generated path."""

    if isinstance(shard_count, bool) or not isinstance(shard_count, int):
        raise TypeError("shard_count must be an integer")
    if shard_count < 1:
        raise ValueError("production shard count must be positive")
    if unit_mode not in ("stable-shards", "class"):
        raise ValueError("unit_mode must be 'stable-shards' or 'class'")
    selections = load_production_kernel_selections(manifest, architecture, profile)
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    if unit_mode == "class":
        selected_by_name = {selection.spec.name: selection for selection in selections}
        names = (
            tuple(sorted(FUSED_SHELL_SPEC_BY_NAME))
            if all_class_units
            else tuple(sorted(selected_by_name))
        )
        units = tuple(
            (name, ((selected_by_name[name],) if name in selected_by_name else ()))
            for name in names
        )
        for name, unit in units:
            path = output_directory / f"vibeqc_generated_shell_{name}.cu"
            _write_if_changed(path, emit_production_shard(unit))
            outputs.append(path)
    else:
        shards = _partition_production_selections(selections, shard_count)
        for index, shard in enumerate(shards):
            path = output_directory / f"vibeqc_generated_shell_shard_{index}.cu"
            _write_if_changed(path, emit_production_shard(shard))
            outputs.append(path)
    header = output_directory / "vibeqc_generated_shell_registry.hpp"
    source = output_directory / "vibeqc_generated_shell_registry.cu"
    _write_if_changed(header, emit_registry_header(selections))
    _write_if_changed(source, emit_registry_source(selections))
    outputs.extend((header, source))
    return tuple(outputs)
