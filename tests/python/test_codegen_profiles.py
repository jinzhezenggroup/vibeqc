"""Focused profile and benchmark-admission tests extracted from the legacy codegen suite.

See issue #489.
"""

from __future__ import annotations

import json
import os
import subprocess
import typing
from pathlib import Path

import pytest
from vibeqc_compiler.integral import (
    DDPS_SPEC,
    DPDS_SPEC,
    FUSED_SHELL_SPEC_BY_NAME,
    PSPS_SPEC,
    KernelConsumer,
    cuda_target_info,
)
from vibeqc_compiler.integral.batch_benchmark import (
    DEFAULT_CANDIDATES,
    benchmark_command,
    candidate_specs,
    discover_candidate_specs,
    emit_batch_driver,
    emit_candidate_translation_unit,
    parse_ptxas_resources,
    rank_profiled_candidates,
)
from vibeqc_compiler.integral.batch_benchmark import (
    _compile_candidate as _compile_batch_candidate,
)
from vibeqc_compiler.integral.benchmark import (
    benchmark_command as standalone_benchmark_command,
)
from vibeqc_compiler.integral.benchmark import emit_shell_class_benchmark_cuda
from vibeqc_compiler.integral.production import (
    resolve_production_profile,
    write_production_bundles,
)

TEST_CUDA_TARGET = cuda_target_info("sm_120")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("architecture", ("sm_80", "sm_86", "sm_89", "sm_90"))
def test_unmeasured_cuda_targets_require_explicit_portable_profile(
    architecture: str,
) -> None:
    """Never hide a missing tuned profile behind an implicit generic build."""

    manifest = (
        REPOSITORY_ROOT
        / "python"
        / "vibeqc_compiler"
        / "integral"
        / "production_shell_classes.json"
    )
    with pytest.raises(ValueError, match="portable_cuda.*explicitly"):
        resolve_production_profile(manifest, architecture)
    resolved = resolve_production_profile(manifest, architecture, "portable_cuda")
    assert resolved.profile == "portable_cuda"
    assert resolved.portable is True
    assert resolved.tuned is False
    assert resolved.selections == ()
    with pytest.raises(ValueError, match="incompatible"):
        resolve_production_profile(manifest, architecture, "sm_120")


def _small_multi_profile_manifest(path: Path) -> None:
    """Write two legal measured profiles for collision/link tests."""

    schedule = {
        "kind": "packed_tasks",
        "block_threads": 32,
        "component_tile": 6,
        "tasks_per_warp": 32,
        "shared_coulomb": False,
        "pair_orientation": "canonical",
        "pair_storage": "materialized",
        "unroll_pair_terms": True,
    }
    profile = {
        "kind": "tuned",
        "cuda_toolkit": "12.9.1",
        "generator_abi": 1,
        "kernels": [
            {
                "shell_class": "dsss",
                "consumers": ["force"],
                "schedule": schedule,
            }
        ],
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "default_architecture": "sm_120",
                "architectures": {
                    "sm_80": profile,
                    "sm_120": profile,
                    "portable_cuda": {"kind": "portable", "kernels": []},
                },
            }
        ),
        encoding="utf-8",
    )


def test_multi_profile_bundle_is_order_independent_and_collision_free(
    tmp_path: Path,
) -> None:
    """Generate separate symbols, metadata, and shards for every target."""

    manifest = tmp_path / "manifest.json"
    _small_multi_profile_manifest(manifest)
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first = write_production_bundles(manifest, first_directory, 1, ("sm_120", "sm_80"))
    second = write_production_bundles(
        manifest, second_directory, 1, ("sm_80", "sm_120")
    )
    assert [path.relative_to(first_directory) for path in first] == [
        path.relative_to(second_directory) for path in second
    ]
    for first_path, second_path in zip(first, second, strict=True):
        assert first_path.read_bytes() == second_path.read_bytes()

    registry = (first_directory / "vibeqc_generated_shell_registry.cu").read_text(
        encoding="utf-8"
    )
    assert "vibeqc_launch_sm80_generated_dsss" in registry
    assert "vibeqc_launch_sm120_generated_dsss" in registry
    header = (first_directory / "vibeqc_generated_shell_registry.hpp").read_text(
        encoding="utf-8"
    )
    assert header.index('"sm_80"') < header.index('"sm_120"')
    sm80 = next(path for path in first if "sm80_shard" in path.name).read_text(
        encoding="utf-8"
    )
    sm120 = next(path for path in first if "sm120_shard" in path.name).read_text(
        encoding="utf-8"
    )
    assert "namespace vibeqc::scf::generated::profile_sm80" in sm80
    assert "namespace vibeqc::scf::generated::profile_sm120" in sm120


def test_multi_profile_objects_compile_and_link_when_nvcc_is_configured(
    tmp_path: Path,
) -> None:
    """Verify two architecture bundles do not collide at host or device link."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the multi-profile compile/link test")
    manifest = tmp_path / "manifest.json"
    _small_multi_profile_manifest(manifest)
    output = tmp_path / "generated"
    write_production_bundles(manifest, output, 1, ("sm_80", "sm_120"))
    objects = []
    for architecture in ("sm_80", "sm_120"):
        source = next(
            (output / architecture).glob(
                f"vibeqc_generated_shell_{architecture.replace('_', '')}_shard_0.cu"
            )
        )
        obj = tmp_path / f"{architecture}.o"
        result = subprocess.run(
            [
                nvcc,
                "-std=c++20",
                f"-arch={architecture}",
                f"-I{REPOSITORY_ROOT / 'src'}",
                "-Xcompiler=-fPIC",
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(obj)
    registry_object = tmp_path / "registry.o"
    result = subprocess.run(
        [
            nvcc,
            "-std=c++20",
            "-arch=sm_80",
            f"-I{output}",
            f"-I{REPOSITORY_ROOT / 'src'}",
            "-Xcompiler=-fPIC",
            "-c",
            str(output / "vibeqc_generated_shell_registry.cu"),
            "-o",
            str(registry_object),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    library = tmp_path / "libprofiles.so"
    result = subprocess.run(
        [
            nvcc,
            "-shared",
            str(registry_object),
            *(map(str, objects)),
            "-o",
            str(library),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_virtual_cuda_target_keeps_host_profile_portable() -> None:
    """Do not apply a measured host schedule to PTX that may JIT on a future GPU."""

    cmake = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    profile_block = cmake.split("# A single real architecture may use", 1)[1].split(
        "vibeqc_register_cuda_generated_sources", 1
    )[0]
    virtual_guard = profile_block.index(
        'if(NOT _vibeqc_cuda_profile_architecture MATCHES "-virtual$")'
    )
    real_normalization = profile_block.index(
        'string(REGEX REPLACE "-real$" "" _vibeqc_cuda_profile_architecture'
    )
    profile_define = profile_block.index("VIBEQC_CUDA_PROFILE_ARCHITECTURE=")
    assert virtual_guard < real_normalization < profile_define
    assert 'REGEX REPLACE "-.*$"' not in profile_block


def test_batch_screening_ranks_real_profile_and_emits_one_process_driver() -> None:
    with pytest.raises(ValueError, match="requires --profile"):
        candidate_specs()

    payload = {
        "shell_classes": [
            {"class": "dppp"},
            {"class": "ppps"},
            {"class": "ddps"},
            {"class": "psss"},
        ]
    }
    ranked = rank_profiled_candidates(payload, limit=2)
    assert tuple(spec.name for spec in ranked) == ("psss",)
    candidate = DEFAULT_CANDIDATES[0]
    source = emit_candidate_translation_unit(
        candidate,
        task_count=2,
        primitive_count=1,
        warmups=0,
        iterations=1,
        samples=1,
        target=TEST_CUDA_TARGET,
    )
    assert f"vibeqc_run_shell_class_{candidate.name}" in source
    driver = emit_batch_driver((candidate,))
    assert "cudaFree(nullptr)" in driver
    assert f"vibeqc_run_shell_class_{candidate.name}()" in driver


def test_batch_screening_discovers_consumer_specific_manifest_gap() -> None:
    """Discover a bounded work-ranked gap without a hand-written name list."""

    force = discover_candidate_specs(consumer=KernelConsumer.FORCE, limit=3)
    assert tuple(spec.name for spec in force) == ("ffff", "fffd", "fdfd")
    fock = discover_candidate_specs(consumer=KernelConsumer.FOCK, limit=3)
    assert tuple(spec.name for spec in fock) == ("ffff", "fffd", "fdfd")
    assert FUSED_SHELL_SPEC_BY_NAME["fpps"] in discover_candidate_specs(
        consumer=KernelConsumer.FOCK
    )


def test_batch_screening_sorts_unsorted_profile_work_and_deduplicates() -> None:
    """Choose f-shell candidates by measured work, not profile row order."""

    payload = {
        "shell_classes": [
            {"class": "fsss", "primitive_quartets": 10},
            {"class": "fsps", "primitive_work": 70},
            {"class": "fddd", "primitive_quartets": 600},
            # A duplicate row can occur when profiles combine orientations.
            {"class": "fsps", "primitive_work": 700},
        ]
    }

    ranked = rank_profiled_candidates(payload, limit=3)

    assert tuple(spec.name for spec in ranked) == ("fsps", "fddd", "fsss")


def test_batch_screening_excludes_production_classes_per_consumer() -> None:
    """Allow a force-only production class to enter the Fock screener."""

    # FPPS is promoted for force but not for coefficient-only Fock.  The
    # explicit and profiled paths must therefore agree that it is a Fock
    # candidate while still rejecting it from force screening.
    with pytest.raises(ValueError, match="fpps.*force production AOT"):
        candidate_specs(("fpps",), consumer=KernelConsumer.FORCE)
    assert tuple(
        spec.name
        for spec in candidate_specs(
            ("fpps", "fpps"),
            consumer=KernelConsumer.FOCK,
        )
    ) == ("fpps",)

    payload = {
        "shell_classes": [
            {"class": "ssss", "primitive_work": 900},
            {"class": "fpps", "primitive_work": 1000},
        ]
    }
    ranked = rank_profiled_candidates(
        payload,
        limit=2,
        consumer=KernelConsumer.FOCK,
    )
    assert tuple(spec.name for spec in ranked) == ("fpps",)


def test_batch_screening_can_emit_coefficient_only_fock_candidates() -> None:
    """Route Fock screening through the same generated task ABI."""

    source = emit_candidate_translation_unit(
        FUSED_SHELL_SPEC_BY_NAME["fsps"],
        task_count=2,
        primitive_count=1,
        warmups=0,
        iterations=1,
        samples=1,
        consumer=KernelConsumer.FOCK,
        target=TEST_CUDA_TARGET,
    )

    assert r"\"consumer\":\"fock\"" in source
    assert "generated_fsps_shell_class_fock_rhf_kernel" in source
    assert r"\"maximum_fock_error\"" in source


def test_batch_benchmark_command_has_finite_slurm_allocation() -> None:
    """Keep manual batch execution aligned with the CUDA adapter contract."""

    command = benchmark_command(
        Path("build/shell_batch_benchmark"),
        srun="srun",
        partition="main",
        gres="gpu:5090:1",
        slurm_time="00:07:00",
    )

    assert command == [
        "srun",
        "--partition=main",
        "--gres=gpu:5090:1",
        "--nodes=1",
        "--ntasks=1",
        "--time=00:07:00",
        "build/shell_batch_benchmark",
    ]
    with pytest.raises(ValueError, match="non-empty"):
        benchmark_command(Path("benchmark"), slurm_time=" ")


def test_standalone_benchmark_command_has_finite_slurm_allocation() -> None:
    """Keep the standalone CUDA benchmark under the same scheduler guard."""

    assert standalone_benchmark_command(
        Path("build/dppp_benchmark"),
        slurm_time="00:05:00",
    ) == [
        "srun",
        "--partition=main",
        "--gres=gpu:1",
        "--nodes=1",
        "--ntasks=1",
        "--time=00:05:00",
        "build/dppp_benchmark",
    ]


def test_ptxas_resource_parser_selects_fock_symbol_family() -> None:
    """Keep Fock resource gates independent from force resource records."""

    diagnostics = (
        "Function properties for generated_fsps_shell_class_fock_rhf_kernel\n"
        "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
        "ptxas info    : Used 64 registers, 0 bytes lmem, 0 bytes smem\n"
    )

    resources = parse_ptxas_resources(
        diagnostics,
        "fsps",
        consumer=KernelConsumer.FOCK,
    )

    assert len(resources) == 1
    assert resources[0].function.endswith("fock_rhf_kernel")


def test_batch_candidate_compile_records_artifact_provenance(
    tmp_path: Path,
) -> None:
    """Keep batch screening reports auditable even with a failed PTXAS parse."""

    source = tmp_path / f"{PSPS_SPEC.name}_candidate.cu"
    source.write_text("// generated source\n", encoding="utf-8")
    fake_nvcc = tmp_path / "fake-nvcc"
    fake_nvcc.write_text(
        """#!/bin/sh
output=''
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then
    shift
    output="$1"
  fi
  shift
done
printf 'fake object' > "$output"
""",
        encoding="utf-8",
    )
    fake_nvcc.chmod(0o755)

    row = _compile_batch_candidate(fake_nvcc, "sm_120", tmp_path, PSPS_SPEC)

    assert row["returncode"] == 0
    assert row["compile_seconds"] >= 0.0
    assert row["source_bytes"] == source.stat().st_size
    assert row["object_bytes"] == len("fake object")


@pytest.mark.parametrize(
    ("spec", "third_offset"),
    ((DPDS_SPEC, 9), (DDPS_SPEC, 12)),
)
def test_benchmark_is_generated_without_shell_specific_harness_code(
    spec: typing.Any, third_offset: typing.Any
) -> None:
    source = emit_shell_class_benchmark_cuda(
        spec,
        task_count=32,
        primitive_count=2,
        warmups=1,
        iterations=3,
        samples=5,
        target=TEST_CUDA_TARGET,
    )
    assert "constexpr std::size_t n = 16U" in source
    assert f"task.ao_begin[2] = {third_offset}U" in source
    assert "task.ao_begin[3] = 15U" in source
    assert f"generated_{spec.name}_component_gradient<true>" in source
    assert f"generated_{spec.name}_component_gradient<false>" in source
    assert f"generated_{spec.name}_component_recompute_rhf_kernel" in source
    assert "generated_dppp" not in source
