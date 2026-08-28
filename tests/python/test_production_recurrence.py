"""Production-manifest plumbing for alternate integral recurrences."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.vibeqc_codegen.production import (
    emit_multi_registry_header,
    emit_multi_registry_source,
    emit_production_shard,
    emit_profile_shard,
    emit_registry_header,
    emit_registry_source,
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


def _rys4_manifest(path: Path) -> None:
    """Write the mixed DPPP row used by production force/Fock dispatch."""

    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "default_architecture": "sm_120",
                "architectures": {
                    "sm_120": {
                        "kernels": [
                            {
                                "shell_class": "dppp",
                                "consumers": ["fock", "force"],
                                "recurrence": "rys4",
                                "schedule": {
                                    "kind": "component_lanes",
                                    "block_threads": 192,
                                    "component_tile": 162,
                                    "tasks_per_warp": 1,
                                    "shared_coulomb": True,
                                    "pair_orientation": "swapped",
                                    "pair_storage": "materialized",
                                    "unroll_pair_terms": True,
                                    "minimum_blocks_per_sm": 2,
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


def test_mixed_dppp_rys4_manifest_emits_rys_force_and_existing_fock(
    tmp_path: Path,
):
    """Use Rys4 only for force while preserving the accepted direct Fock."""

    manifest = tmp_path / "rys4.json"
    _rys4_manifest(manifest)
    resolved = resolve_production_profile(manifest, "sm_120")
    selection = resolved.selections[0]
    assert selection.recurrence == "rys4"
    assert [consumer.value for consumer in selection.consumers] == [
        "fock",
        "force",
    ]
    shard = emit_production_shard(resolved.selections)
    assert "generated_dppp_rys4_component_lane_task" in shard
    assert "generated_dppp_shell_class_fock_rhf_kernel" in shard
    profile_shard = emit_profile_shard(resolved, resolved.selections)
    assert "generated_sm120_dppp_rys4_component_lane_task" in profile_shard
    assert "generated_sm120_dppp_shell_class_fock_rhf_kernel" in profile_shard


def test_rys4_manifest_rejects_noncooperative_schedule(tmp_path: Path):
    """Fail at the manifest boundary instead of inside the CUDA emitter."""

    manifest = tmp_path / "rys4_thread_tasks.json"
    _rys4_manifest(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    schedule = payload["architectures"]["sm_120"]["kernels"][0]["schedule"]
    schedule.update(
        {
            "kind": "thread_tasks",
            "block_threads": 192,
            "tasks_per_warp": 32,
            "shared_coulomb": False,
        }
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="dppp component-lane schedule"):
        load_production_kernel_selections(manifest, "sm_120")


def test_existing_production_rows_default_to_subset_wick():
    """Keep unmodified rows on subset/Wick beside the promoted DPPP force."""

    repository_root = Path(__file__).resolve().parents[2]
    manifest = (
        repository_root
        / "tools"
        / "vibeqc_codegen"
        / "production_shell_classes.json"
    )
    selections = load_production_kernel_selections(manifest, "sm_120")
    assert selections
    assert next(
        selection for selection in selections if selection.spec.name == "dppp"
    ).recurrence == "rys4"
    assert all(
        selection.recurrence == "subset_wick"
        for selection in selections
        if selection.spec.name != "dppp"
    )
    assert next(
        selection for selection in selections if selection.spec.name == "ppps"
    ).resident_force_recurrence == "rys3"


def test_ppps_resident_option_keeps_ordinary_fock_force_fallback(tmp_path: Path):
    """Emit resident Rys3 beside, rather than instead of, ppps force/Fock."""

    manifest = tmp_path / "resident.json"
    row = {
        "shell_class": "ppps",
        "consumers": ["fock", "force"],
        "resident_force_recurrence": "rys3",
        "schedule": {
            "kind": "component_lanes",
            "block_threads": 32,
            "component_tile": 27,
            "tasks_per_warp": 1,
            "shared_coulomb": True,
            "pair_orientation": "canonical",
            "pair_storage": "materialized",
            "unroll_pair_terms": True,
        },
    }
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "default_architecture": "sm_120",
                "architectures": {"sm_120": {"kernels": [row]}},
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_production_profile(manifest, "sm_120")
    selection = resolved.selections[0]
    assert selection.recurrence == "subset_wick"
    assert selection.resident_force_recurrence == "rys3"

    source = emit_production_shard(resolved.selections)
    assert "vibeqc_launch_generated_ppps(" in source
    assert "vibeqc_launch_generated_ppps_fock(" in source
    assert 'extern "C" cudaError_t vibeqc_launch_ppps_resident(' in source
    assert "generated_ppps_resident_bra_force_rhf_kernel<<<" in source
    assert "generated_ppps_resident_bra_force_uhf_kernel<<<" in source
    assert "const void* resident_tasks" in source
    assert "const void* ket_tasks" in source
    assert "std::size_t task_count" in source
    assert "ket_base += kGeneratedPppsResidentBlockThreads" in source
    assert "resident.ket_count > kGeneratedPppsResidentBlockThreads" not in source

    profile_source = emit_profile_shard(resolved, resolved.selections)
    assert "vibeqc::scf::detail::GeneratedPppsResidentTask" in profile_source
    assert "vibeqc::scf::detail::GeneratedSm120PppsResidentTask" not in profile_source

    registry_header = emit_registry_header(resolved.selections)
    registry_source = emit_registry_source(resolved.selections)
    assert "launch_ppps_resident" in registry_header
    assert "vibeqc_launch_ppps_resident" in registry_source
    assert "return cudaErrorNotSupported;" not in registry_source


def test_ppps_resident_registry_falls_back_when_not_selected(tmp_path: Path):
    """Keep the API safe for profiles that do not compile a resident route."""

    manifest = tmp_path / "ordinary.json"
    _rys3_manifest(manifest, ["force"])
    resolved = resolve_production_profile(manifest, "sm_120")
    registry_source = emit_registry_source(resolved.selections)
    assert "launch_ppps_resident" in emit_registry_header(resolved.selections)
    assert "return cudaErrorNotSupported;" in registry_source


def test_multi_profile_resident_registry_tracks_each_profile(tmp_path: Path):
    """Select the resident function pointer together with the CUDA profile."""

    schedule = {
        "kind": "component_lanes",
        "block_threads": 32,
        "component_tile": 27,
        "tasks_per_warp": 1,
        "shared_coulomb": True,
        "pair_orientation": "canonical",
        "pair_storage": "materialized",
        "unroll_pair_terms": True,
    }

    def profile(resident: bool) -> dict[str, object]:
        row: dict[str, object] = {
            "shell_class": "ppps",
            "consumers": ["fock", "force"],
            "schedule": schedule,
        }
        if resident:
            row["resident_force_recurrence"] = "rys3"
        return {"kernels": [row]}

    manifest = tmp_path / "multi.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "default_architecture": "sm_120",
                "architectures": {
                    "sm_80": profile(False),
                    "sm_120": profile(True),
                },
            }
        ),
        encoding="utf-8",
    )
    sm80 = resolve_production_profile(manifest, "sm_80")
    sm120 = resolve_production_profile(manifest, "sm_120")
    source = emit_multi_registry_source((sm80, sm120))
    header = emit_multi_registry_header((sm80, sm120))
    assert '#include "scf/aot_shell_registry.hpp"' in header
    assert "launch_sm80_resident" in source
    assert "return cudaErrorNotSupported;" in source
    assert "launch_sm120_resident" in source
    assert "vibeqc_launch_sm120_ppps_resident" in source
