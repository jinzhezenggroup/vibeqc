"""Production-manifest plumbing for alternate integral recurrences."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.vibeqc_codegen.production import (
    emit_production_shard,
    emit_profile_shard,
    load_production_kernel_selections,
    resolve_production_profile,
)


def _rys3_manifest(path: Path, consumers: list[str]) -> None:
    """Write the smallest v2 manifest that exercises the ppps Rys path."""

    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "default_architecture": "sm_120",
                "architectures": {
                    "sm_120": {
                        "kernels": [
                            {
                                "shell_class": "ppps",
                                "consumers": consumers,
                                "recurrence": "rys3",
                                "schedule": {
                                    "kind": "thread_tasks",
                                    "block_threads": 32,
                                    "component_tile": 27,
                                    "tasks_per_warp": 32,
                                    "shared_coulomb": False,
                                    "minimum_blocks_per_sm": 8,
                                },
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_force_only_rys3_manifest_reaches_every_production_emitter(
    tmp_path: Path,
):
    """Keep the row-level recurrence when shards and profiles are emitted."""

    manifest = tmp_path / "rys3.json"
    _rys3_manifest(manifest, ["force"])

    selections = load_production_kernel_selections(manifest, "sm_120")
    assert len(selections) == 1
    assert selections[0].recurrence == "rys3"

    shard = emit_production_shard(selections)
    assert "generated_ppps_rys3_force_task" in shard

    resolved = resolve_production_profile(manifest, "sm_120")
    profile_shard = emit_profile_shard(resolved, resolved.selections)
    assert "generated_sm120_ppps_rys3_force_task" in profile_shard


def test_rys3_rejects_a_fock_consumer_at_manifest_boundary(tmp_path: Path):
    """Reject unsupported mixed Rys/Fock rows before CUDA source generation."""

    manifest = tmp_path / "rys3_fock.json"
    _rys3_manifest(manifest, ["fock", "force"])

    with pytest.raises(ValueError, match="force-only"):
        load_production_kernel_selections(manifest, "sm_120")


def test_existing_production_rows_default_to_subset_wick():
    """Keep old manifests source-compatible without editing their rows."""

    repository_root = Path(__file__).resolve().parents[2]
    manifest = (
        repository_root
        / "tools"
        / "vibeqc_codegen"
        / "production_shell_classes.json"
    )
    selections = load_production_kernel_selections(manifest, "sm_120")
    assert selections
    assert all(selection.recurrence == "subset_wick" for selection in selections)
