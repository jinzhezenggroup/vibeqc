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


def _structural_rys_manifest(
    path: Path,
    *,
    shell_class: str,
    recurrence: str,
    schedule: dict[str, object],
) -> None:
    """Write one force-only row for recurrence capability-boundary tests."""

    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "default_architecture": "sm_120",
                "architectures": {
                    "sm_120": {
                        "kernels": [
                            {
                                "shell_class": shell_class,
                                "consumers": ["force"],
                                "recurrence": recurrence,
                                "schedule": schedule,
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

    with pytest.raises(ValueError, match="requires fock_schedule"):
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

    with pytest.raises(ValueError, match="production rys4 requires supported"):
        load_production_kernel_selections(manifest, "sm_120")


def test_scalar_rys3_accepts_structurally_legal_f_shell(tmp_path: Path):
    """Accept FSSS from root count and scalar-backend shape, not a name list."""

    manifest = tmp_path / "fsss_scalar_rys3.json"
    _structural_rys_manifest(
        manifest,
        shell_class="fsss",
        recurrence="rys3",
        schedule={
            "kind": "thread_tasks",
            "block_threads": 32,
            "component_tile": 10,
            "tasks_per_warp": 32,
            "shared_coulomb": False,
            "minimum_blocks_per_sm": 1,
        },
    )

    resolved = resolve_production_profile(manifest, "sm_120")
    selection = resolved.selections[0]
    assert selection.spec.name == "fsss"
    assert selection.recurrence == "rys3"
    shard = emit_profile_shard(resolved, resolved.selections)
    assert "generated_sm120_fsss_rys3_force_task" in shard


def test_uniform_warp_rys4_accepts_structurally_legal_f_shell(tmp_path: Path):
    """Allow an f-shell Rys4 program when its mapping needs no d-only decoder."""

    manifest = tmp_path / "fpps_uniform_rys4.json"
    _structural_rys_manifest(
        manifest,
        shell_class="fpps",
        recurrence="rys4",
        schedule={
            "kind": "subgroup_tasks",
            "block_threads": 256,
            "component_tile": 90,
            "tasks_per_warp": 4,
            "shared_coulomb": True,
            "minimum_blocks_per_sm": 1,
        },
    )

    resolved = resolve_production_profile(manifest, "sm_120")
    selection = resolved.selections[0]
    assert selection.spec.name == "fpps"
    assert selection.recurrence == "rys4"
    shard = emit_profile_shard(resolved, resolved.selections)
    assert "generated_sm120_fpps_rys4_uniform_warp_batch" in shard


def test_component_lane_rys4_rejects_f_shell_decoder_gap(tmp_path: Path):
    """Keep the current runtime-indexed s/p/d table limit structural."""

    manifest = tmp_path / "fpps_component_rys4.json"
    _structural_rys_manifest(
        manifest,
        shell_class="fpps",
        recurrence="rys4",
        schedule={
            "kind": "component_lanes",
            "block_threads": 96,
            "component_tile": 90,
            "tasks_per_warp": 1,
            "shared_coulomb": True,
            "minimum_blocks_per_sm": 1,
        },
    )

    with pytest.raises(ValueError, match="production rys4 requires supported"):
        load_production_kernel_selections(manifest, "sm_120")


def test_manifest_rys_root_count_comes_from_integral_ir(tmp_path: Path):
    """Reject a fixed-root count mismatch before considering CUDA mapping."""

    manifest = tmp_path / "fsss_wrong_roots.json"
    _structural_rys_manifest(
        manifest,
        shell_class="fsss",
        recurrence="rys2",
        schedule={
            "kind": "thread_tasks",
            "block_threads": 32,
            "component_tile": 10,
            "tasks_per_warp": 32,
            "shared_coulomb": False,
        },
    )

    with pytest.raises(ValueError, match="fsss.*requires rys3, not rys2"):
        load_production_kernel_selections(manifest, "sm_120")


def test_existing_production_rows_default_to_subset_wick():
    """Keep only explicitly promoted force rows on fixed-root Rys."""

    repository_root = Path(__file__).resolve().parents[2]
    manifest = (
        repository_root / "tools" / "vibeqc_codegen" / "production_shell_classes.json"
    )
    selections = load_production_kernel_selections(manifest, "sm_120")
    assert selections
    recurrences = {
        selection.spec.name: selection.recurrence for selection in selections
    }
    assert all(
        recurrences[name] == "rys4"
        for name in ("dppp", "dpdp", "dpds", "ddpp", "ddps", "ddds")
    )
    assert all(
        recurrences[name] == "rys3"
        for name in (
            "ppps",
            "dpps",
            "dpss",
            "dsps",
            "dspp",
            "dsds",
            "ddss",
            "pppp",
        )
    )
    assert all(recurrences[name] == "rys2" for name in ("psps", "ppss", "dsss"))
    assert all(recurrences[name] == "rys5" for name in ("dddp", "dddd"))
    assert all(
        selection.recurrence == "subset_wick"
        for selection in selections
        if selection.spec.name
        not in {
            "dppp",
            "dpdp",
            "dpds",
            "ddpp",
            "ddps",
            "ddds",
            "ppps",
            "dpps",
            "dpss",
            "dsps",
            "dspp",
            "dsds",
            "ddss",
            "pppp",
            "psps",
            "ppss",
            "dsss",
            "dddp",
            "dddd",
        }
    )
    assert (
        next(
            selection for selection in selections if selection.spec.name == "ppps"
        ).resident_force_recurrence
        == "rys3"
    )


@pytest.mark.parametrize("shell_class", ("dsps", "dpps"))
def test_three_root_classes_accept_shared_scalar_thread_backend(
    tmp_path: Path, shell_class: str
):
    """Prove that one generator-level scalar Rys3 path covers both classes."""

    component_count = {"dsps": 18, "dpps": 54}[shell_class]
    manifest = tmp_path / f"{shell_class}_scalar_rys3.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "default_architecture": "sm_120",
                "architectures": {
                    "sm_120": {
                        "kernels": [
                            {
                                "shell_class": shell_class,
                                "consumers": ["fock", "force"],
                                "recurrence": "rys3",
                                "schedule": {
                                    "kind": "thread_tasks",
                                    "block_threads": 32,
                                    "component_tile": component_count,
                                    "tasks_per_warp": 32,
                                    "shared_coulomb": False,
                                    "minimum_blocks_per_sm": 1,
                                },
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_production_profile(manifest, "sm_120")
    shard = emit_profile_shard(resolved, resolved.selections)
    assert f"generated_sm120_{shell_class}_rys3_force_task" in shard
    assert "generated_sm120_ppps_rys3_force_task" not in shard
    assert f"generated_sm120_{shell_class}_shell_class_fock_rhf_kernel" in shard


def test_production_dsps_promotes_scalar_force_but_retains_component_fock():
    """Keep the measured force promotion independent of the value consumer."""

    repository_root = Path(__file__).resolve().parents[2]
    manifest = (
        repository_root / "tools" / "vibeqc_codegen" / "production_shell_classes.json"
    )
    resolved = resolve_production_profile(manifest, "sm_120")
    selection = next(item for item in resolved.selections if item.spec.name == "dsps")
    assert selection.schedule.kind.value == "thread_tasks"
    shard = emit_profile_shard(resolved, (selection,))
    assert "generated_sm120_dsps_rys3_force_task" in shard
    assert "generated_sm120_dsps_shell_class_fock_rhf_kernel" in shard
    assert "generated_sm120_dsps_scalar_thread_fock" not in shard


@pytest.mark.parametrize(
    ("shell_class", "fock_block_threads"),
    (("ppps", 128), ("dpps", 128), ("pppp", 128)),
)
def test_production_rys3_uniform_force_keeps_independent_fock_schedule(
    shell_class: str, fock_block_threads: int
):
    """Allow each accepted value path to use its independently tuned mapping."""

    repository_root = Path(__file__).resolve().parents[2]
    manifest = (
        repository_root / "tools" / "vibeqc_codegen" / "production_shell_classes.json"
    )
    resolved = resolve_production_profile(manifest, "sm_120")
    selection = next(
        item for item in resolved.selections if item.spec.name == shell_class
    )
    assert selection.schedule.kind.value == "subgroup_tasks"
    assert selection.schedule.tasks_per_block == 32
    shard = emit_profile_shard(resolved, (selection,))
    class_name = shell_class[0].upper() + shell_class[1:]
    assert f"kGeneratedSm120{class_name}Rys3TaskCount = 32U" in shard
    assert f"generated_sm120_{shell_class}_shell_class_fock_rhf_kernel" in shard
    assert (
        f"kGeneratedSm120{class_name}FockBlockThreads = {fock_block_threads}U" in shard
    )


@pytest.mark.parametrize(
    (
        "shell_class",
        "force_schedule",
        "force_block_threads",
        "fock_schedule",
        "fock_block_threads",
        "explicit_fock_schedule",
    ),
    (
        ("dpdp", "subgroup_tasks", 256, "component_lanes", 352, True),
        ("dpds", "subgroup_tasks", 256, "subgroup_tasks", 128, True),
        ("ddpp", "subgroup_tasks", 256, "component_lanes", 352, True),
        ("ddps", "subgroup_tasks", 256, "subgroup_tasks", 128, True),
        ("ddds", "component_lanes", 224, "subgroup_tasks", 128, True),
    ),
)
def test_production_rys4_force_retains_explicit_fock_schedule(
    shell_class: str,
    force_schedule: str,
    force_block_threads: int,
    fock_schedule: str,
    fock_block_threads: int,
    explicit_fock_schedule: bool,
):
    """Keep each accepted Rys4 Fock mapping explicit or intentionally shared."""

    repository_root = Path(__file__).resolve().parents[2]
    manifest = (
        repository_root / "tools" / "vibeqc_codegen" / "production_shell_classes.json"
    )
    resolved = resolve_production_profile(manifest, "sm_120")
    selection = next(
        item for item in resolved.selections if item.spec.name == shell_class
    )
    assert selection.recurrence == "rys4"
    assert selection.schedule.kind.value == force_schedule
    assert selection.schedule.block_threads == force_block_threads
    assert (selection.fock_schedule is not None) == explicit_fock_schedule
    effective_fock_schedule = selection.fock_schedule or selection.schedule
    assert effective_fock_schedule.kind.value == fock_schedule
    assert effective_fock_schedule.block_threads == fock_block_threads
    if shell_class in {"dpdp", "ddpp"}:
        assert selection.schedule.pair_storage.value == "materialized"
        assert effective_fock_schedule.pair_storage.value == "recomputed"
        assert effective_fock_schedule.minimum_blocks_per_sm == 1

    shard = emit_profile_shard(resolved, (selection,))
    class_name = shell_class[0].upper() + shell_class[1:]
    assert f"kGeneratedSm120{class_name}BlockThreads = {force_block_threads}U" in shard
    assert (
        f"kGeneratedSm120{class_name}FockBlockThreads = {fock_block_threads}U" in shard
    )
    assert f"generated_sm120_{shell_class}_rys4" in shard
    assert f"generated_sm120_{shell_class}_shell_class_fock_rhf_kernel" in shard
    fock_launch = shard.split(
        f'extern "C" cudaError_t vibeqc_launch_sm120_generated_{shell_class}_fock(',
        maxsplit=1,
    )[1].split('extern "C"', maxsplit=1)[0]
    assert (
        f"worker_blocks, kGeneratedSm120{class_name}FockBlockThreads, 0, stream"
        in fock_launch
    )
    assert (
        f"worker_blocks, kGeneratedSm120{class_name}BlockThreads, 0, stream"
        not in fock_launch
    )


def test_production_dddp_rys5_retains_explicit_fock_schedule():
    """Keep the measured DDDP value worker independent of Rys5 force."""

    repository_root = Path(__file__).resolve().parents[2]
    manifest = (
        repository_root / "tools" / "vibeqc_codegen" / "production_shell_classes.json"
    )
    resolved = resolve_production_profile(manifest, "sm_120")
    selection = next(item for item in resolved.selections if item.spec.name == "dddp")
    assert selection.recurrence == "rys5"
    assert selection.schedule.kind.value == "subgroup_tasks"
    assert selection.schedule.block_threads == 256
    assert selection.fock_schedule is not None
    assert selection.fock_schedule.kind.value == "subgroup_tasks"
    assert selection.fock_schedule.block_threads == 128

    shard = emit_profile_shard(resolved, (selection,))
    assert "generated_sm120_dddp_rys5_uniform_warp_batch" in shard
    assert "kGeneratedSm120DddpBlockThreads = 256U" in shard
    assert "kGeneratedSm120DddpFockBlockThreads = 128U" in shard


def test_production_dddd_rys5_retains_native_fock_schedule():
    """Promote only DDDD force while preserving its accepted value worker."""

    repository_root = Path(__file__).resolve().parents[2]
    manifest = (
        repository_root / "tools" / "vibeqc_codegen" / "production_shell_classes.json"
    )
    resolved = resolve_production_profile(manifest, "sm_120")
    selection = next(item for item in resolved.selections if item.spec.name == "dddd")
    assert selection.recurrence == "rys5"
    assert selection.schedule.kind.value == "subgroup_tasks"
    assert selection.schedule.block_threads == 256
    assert selection.fock_schedule is not None
    assert selection.fock_schedule.kind.value == "tiled_components"
    assert selection.fock_schedule.block_threads == 64

    shard = emit_profile_shard(resolved, (selection,))
    assert "generated_sm120_dddd_rys5_uniform_warp_batch" in shard
    assert "kGeneratedSm120DdddBlockThreads = 256U" in shard
    assert "kGeneratedSm120DdddFockBlockThreads = 64U" in shard


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
    assert "unsigned block_threads" in source
    assert "std::size_t task_count" in source
    assert "ket_base += blockDim.x" in source
    assert "block_threads != 32U" in source
    assert "block_threads, 0, stream" in source
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


def test_multi_profile_registry_dispatches_dpps_mixed_fock(tmp_path: Path):
    """Carry the mixed capability and wrapper through profile namespacing."""

    manifest = tmp_path / "mixed.json"
    schedule = {
        "kind": "component_lanes",
        "block_threads": 128,
        "component_tile": 54,
        "tasks_per_warp": 1,
        "shared_coulomb": True,
        "pair_orientation": "canonical",
        "pair_storage": "materialized",
        "unroll_pair_terms": True,
    }
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "default_architecture": "sm_120",
                "architectures": {
                    "sm_120": {
                        "kernels": [
                            {
                                "shell_class": "dpps",
                                "consumers": ["fock", "force"],
                                "schedule": schedule,
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    resolved = resolve_production_profile(manifest, "sm_120")
    shard = emit_profile_shard(resolved, resolved.selections)
    source = emit_multi_registry_source((resolved,))

    assert "vibeqc_launch_sm120_generated_dpps_mixed_fock" in shard
    assert "generated_sm120_dpps_shell_class_mixed_fock_rhf" in shard
    assert "generated_sm120_dpps_shell_class_mixed_fock_task" in shard
    assert "mixed_precision_enabled" in shard
    assert "fp64_threshold" in shard
    assert "stream_state == 3U" in shard
    assert "launch_sm120_mixed_fock" in source
    assert "UINT64_C(2048)" in source
    assert "VIBEQC_AOT_MIXED_FOCK_SHELL_CLASSES" in source
    assert "enabled_mixed_fock_shell_class_mask" in source
    assert "launch_shell_class_mixed_fock" in source


def test_multi_profile_registry_keeps_fock_only_force_symbols_dormant(
    tmp_path: Path,
):
    """Do not make fixed topology reserve force tasks for Fock-only rows."""

    manifest = tmp_path / "fock-only.json"
    schedule = {
        "kind": "packed_tasks",
        "block_threads": 32,
        "component_tile": 1,
        "tasks_per_warp": 32,
        "shared_coulomb": False,
        "pair_orientation": "canonical",
        "pair_storage": "materialized",
        "unroll_pair_terms": True,
    }
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "default_architecture": "sm_120",
                "architectures": {
                    "sm_120": {
                        "kernels": [
                            {
                                "shell_class": "ssss",
                                "consumers": ["fock"],
                                "schedule": schedule,
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    resolved = resolve_production_profile(manifest, "sm_120")
    source = emit_multi_registry_source((resolved,))

    # Bounded spd force dispatch can call the exact dormant symbol directly.
    assert "vibeqc_launch_sm120_generated_ssss" in source
    assert "case 0U:" in source
    # Ordinary force selection and allocation must nevertheless remain empty.
    assert "std::array<ShellKernelMetadata, 0> kForceNames0" in source
    assert "UINT64_C(0)" in source
    assert "std::array<ShellKernelMetadata, 1> kFockNames0" in source
