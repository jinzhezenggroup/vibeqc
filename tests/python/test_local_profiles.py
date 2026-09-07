"""Local tuning safety contracts without requiring CUDA hardware in PR CI."""

import copy
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from vibeqc import profiles
from vibeqc.autotune import endpoint_gate, rank_hotspots, read_xyz


@pytest.fixture
def probe():
    return {
        "source_identity": "a" * 64,
        "native_abi": 1,
        "schema": profiles.PROFILE_SCHEMA,
        "policy": profiles.POLICY,
        "device": {
            "major": 12,
            "minor": 0,
            "name": "test-only GPU",
            "sm_count": 170,
            "registers_per_sm": 65536,
            "shared_memory_per_block": 49152,
            "runtime_version": 12090,
            "toolkit_version": 12090,
            "driver_version": 13000,
            "release_build": 1,
            "fast_compile": 0,
            "portable": 1,
            "official_profile": "portable_cuda",
        },
    }


@pytest.fixture
def bundle(tmp_path, probe):
    """Synthetic byte artifacts exercise storage only; no fake GPU acceptance is published."""
    directory = tmp_path / "fixture"
    directory.mkdir()
    schedule = {"kind": "shell_task", "block_threads": 128}
    kernel = {
        "shell_class": "pppp",
        "consumer": "force",
        "schedule": schedule,
        "schedule_hash": profiles.canonical_hash(schedule),
        "source_hash": "b" * 64,
        "gates": {
            k: "pass"
            for k in ("target", "resources", "numerical", "performance", "endpoint")
        },
    }
    evidence = {
        "endpoint": {"passed": True},
        "candidates": [
            {
                "shell_class": "pppp",
                "consumer": "force",
                "accepted": True,
                "selected_trial": "fixture-only",
                "tuning": {
                    "winners": [
                        {
                            "trial_key": "fixture-only",
                            "schedule": schedule,
                            "production_validation": {"accepted": True},
                        }
                    ]
                },
                "isolated": {
                    "passed": True,
                    "source_hash": "b" * 64,
                    "schedule_hash": profiles.canonical_hash(schedule),
                    "device": {
                        "name": probe["device"]["name"],
                        "major": 12,
                        "minor": 0,
                        "runtime": 12090,
                        "driver": 13000,
                    },
                    "errors": {"fixture": {"passed": True}},
                },
                "endpoint": {"passed": True},
            }
        ],
    }
    (directory / "libvibeqc.so").write_bytes(b"unit-test artifact, not executable")
    profiles.atomic_json(directory / "manifest.json", {})
    profiles.atomic_json(directory / "evidence.json", evidence)
    profile = {
        "schema": "vibeqc.local_profile",
        "schema_version": profiles.PROFILE_SCHEMA,
        "identity": profiles.compatibility_identity(probe),
        "toolchain": {"nvcc": "test NVCC", "ptxas": "test PTXAS"},
        "kernels": [kernel],
        "artifacts": {
            name: profiles.file_hash(directory / name)
            for name in ("libvibeqc.so", "manifest.json", "evidence.json")
        },
    }
    profiles.atomic_json(directory / "profile.json", profile)
    return directory


@pytest.mark.parametrize(
    "field,value",
    [
        ("major", 13),
        ("sm_count", 171),
        ("registers_per_sm", 131072),
        ("shared_memory_per_block", 65536),
        ("runtime_version", 13000),
        ("toolkit_version", 13000),
        ("driver_version", 13001),
        ("name", "another device"),
    ],
)
def test_hardware_and_toolchain_changes_invalidate_profiles(
    bundle, probe, field, value
):
    profiles.validate_bundle(bundle, probe)
    changed = copy.deepcopy(probe)
    changed["device"][field] = value
    with pytest.raises(ValueError, match="identity changed"):
        profiles.validate_bundle(bundle, changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_identity", "c" * 64),
        ("native_abi", 2),
        ("schema", 2),
        ("policy", {"precision": "fp32"}),
    ],
)
def test_source_schema_abi_and_precision_changes_invalidate(
    bundle, probe, field, value
):
    changed = copy.deepcopy(probe)
    changed[field] = value
    with pytest.raises(ValueError, match="identity changed"):
        profiles.validate_bundle(bundle, changed)


def test_nvcc_and_ptxas_are_independently_checked(bundle, probe):
    for name in ("nvcc", "ptxas"):
        tools = {"nvcc": "test NVCC", "ptxas": "test PTXAS", name: "changed"}
        with pytest.raises(ValueError, match="NVCC/PTXAS"):
            profiles.validate_bundle(bundle, probe, tools)


def test_relabeling_foreign_gpu_evidence_does_not_make_it_compatible(bundle, probe):
    evidence_path = bundle / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["candidates"][0]["isolated"]["device"]["major"] = 13
    profiles.atomic_json(evidence_path, evidence)
    profile_path = bundle / "profile.json"
    profile = json.loads(profile_path.read_text())
    profile["artifacts"]["evidence.json"] = profiles.file_hash(evidence_path)
    profiles.atomic_json(profile_path, profile)
    with pytest.raises(ValueError, match="different CUDA target"):
        profiles.validate_bundle(bundle, probe)


def test_cached_binary_must_have_a_tuned_profile_on_the_actual_device(
    probe, tmp_path, monkeypatch
):
    binary = object()
    monkeypatch.setattr(profiles.ctypes, "CDLL", lambda *a: binary)
    monkeypatch.setattr(profiles, "probe_device", lambda *a: probe)
    with pytest.raises(ValueError, match="no tuned profile"):
        profiles.verify_library(tmp_path, probe)
    selected = copy.deepcopy(probe)
    selected["device"]["portable"] = 0
    monkeypatch.setattr(profiles, "probe_device", lambda *a: selected)
    assert profiles.verify_library(tmp_path, probe) is binary


def test_corrupt_and_ungated_profiles_are_rejected(bundle, probe):
    path = bundle / "profile.json"
    original = json.loads(path.read_text())
    for stage in ("target", "resources", "numerical", "performance", "endpoint"):
        changed = copy.deepcopy(original)
        changed["kernels"][0]["gates"][stage] = "not-run"
        profiles.atomic_json(path, changed)
        with pytest.raises(ValueError, match="promotion gates"):
            profiles.validate_bundle(bundle, probe)
    profiles.atomic_json(path, original)
    (bundle / "libvibeqc.so").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact hash"):
        profiles.validate_bundle(bundle, probe)


def test_schedule_and_evidence_hashes_cannot_be_detached(bundle, probe):
    path = bundle / "profile.json"
    payload = json.loads(path.read_text())
    payload["kernels"][0]["schedule"]["block_threads"] = 64
    profiles.atomic_json(path, payload)
    with pytest.raises(ValueError, match="schedule hash"):
        profiles.validate_bundle(bundle, probe)
    payload["kernels"][0]["schedule_hash"] = profiles.canonical_hash(
        payload["kernels"][0]["schedule"]
    )
    payload["kernels"][0]["source_hash"] = "c" * 64
    profiles.atomic_json(path, payload)
    with pytest.raises(ValueError, match="matching independent"):
        profiles.validate_bundle(bundle, probe)


def test_atomic_failure_preserves_previous_profile_and_removes_temporary(
    tmp_path, monkeypatch
):
    path = tmp_path / "active.json"
    profiles.atomic_json(path, {"known-good": "old"})

    def interrupted(*args):
        raise OSError("interrupted replacement")

    monkeypatch.setattr(profiles.os, "replace", interrupted)
    with pytest.raises(OSError):
        profiles.atomic_json(path, {"partial": "new"})
    assert json.loads(path.read_text()) == {"known-good": "old"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["active.json"]


def test_export_import_roundtrip_and_clear_leave_live_binary(
    bundle, probe, tmp_path, monkeypatch
):
    # Only the native-loader boundary is mocked; file validation, copying,
    # immutable bundle selection, locking, and atomic publication run normally.
    monkeypatch.setattr(profiles, "verify_library", lambda *a: object())
    archive = tmp_path / "cluster.zip"
    profiles.export_bundle(bundle, archive)
    root = tmp_path / "cache"
    installed = profiles.import_bundle(archive, probe, root=root)
    assert profiles.validate_bundle(installed, probe) == profiles.validate_bundle(
        bundle, probe
    )
    assert len(json.loads((root / "active.json").read_text())) == 1
    profiles.clear_profiles(root=root)
    assert json.loads((root / "active.json").read_text()) == {}
    assert (installed / "libvibeqc.so").exists()


def test_import_path_traversal_is_rejected(tmp_path, probe):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../libvibeqc.so", "invalid")
    with pytest.raises(ValueError, match="exactly the four"):
        profiles.import_bundle(archive, probe, root=tmp_path / "cache")
    assert not (tmp_path / "libvibeqc.so").exists()


def test_automatic_selection_and_corruption_fall_back(
    bundle, probe, tmp_path, monkeypatch
):
    base, local = object(), object()
    monkeypatch.setenv("VIBEQC_PROFILE_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("VIBEQC_PROFILE", raising=False)
    monkeypatch.setattr(profiles, "probe_device", lambda *a: probe)
    monkeypatch.setattr(profiles, "find_nvcc", lambda: None)
    monkeypatch.setattr(profiles, "verify_library", lambda *a: local)
    installed = profiles.install_bundle(bundle, probe)
    selected, diagnostic = profiles.select_library(base)
    assert selected is local and diagnostic["source"] == "local"
    (installed / "libvibeqc.so").write_bytes(b"corrupt")
    selected, diagnostic = profiles.select_library(base)
    assert selected is base and diagnostic["source"] == "portable"
    assert "artifact hash" in diagnostic["rejected"][0]


def test_hotspots_are_measured_bounded_and_ignore_absent_f_work():
    rows = [
        {"shell_angular": angular, "primitive_quartets": work}
        for angular, work in [
            ((0, 0, 0, 0), 2),
            ((1, 1, 1, 1), 95),
            ((3, 3, 3, 3), 0),
            ((1, 0, 0, 0), 3),
        ]
    ]
    assert [r["name"] for r in rank_hotspots(rows, coverage=0.95)] == ["pppp"]
    selected = rank_hotspots(rows, coverage=1, maximum_classes=2)
    assert [r["name"] for r in selected] == ["pppp", "psss"]
    assert selected[-1]["cumulative_fraction"] == 0.98
    assert rank_hotspots([], coverage=1) == []


def sample(seconds, *, iterations=1):
    return {
        "seconds": seconds,
        "iterations": [iterations],
        "converged": True,
        "backend": ["cuda"],
        "energies": [-1.0],
        "forces": [[[0.0, 0.0, 0.0]]],
    }


def test_endpoint_rejects_slow_noisy_changed_branch_and_wrong_forces():
    baseline = [sample(1.0) for _ in range(6)]
    faster = [sample(0.8) for _ in range(6)]
    assert endpoint_gate(baseline, faster)["passed"]
    assert not endpoint_gate(baseline, [sample(1.1) for _ in range(6)])["passed"]
    assert not endpoint_gate(
        baseline, [sample(t) for t in (0.6, 0.7, 0.8, 1.2, 1.3, 1.4)]
    )["passed"]
    assert not endpoint_gate(baseline, [sample(0.8, iterations=2) for _ in range(6)])[
        "passed"
    ]
    wrong = copy.deepcopy(faster)
    wrong[0]["forces"][0][0][0] = 1e-4
    assert not endpoint_gate(baseline, wrong)["passed"]
    wrong[0]["forces"][0][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        endpoint_gate(baseline, wrong)


def test_xyz_units_and_incomplete_input(tmp_path):
    path = tmp_path / "h2.xyz"
    path.write_text("2\nAngstrom test\nH 0 0 0\nH 0 0 0.529177210903\n")
    np.testing.assert_allclose(read_xyz(path)[1][1], [0, 0, 1])
    path.write_text("2\ntruncated\nH 0 0 0\n")
    with pytest.raises(ValueError, match="complete"):
        read_xyz(path)


def test_installed_cli_help_and_show_need_no_native_library(tmp_path):
    import os

    env = {**os.environ, "VIBEQC_PROFILE_CACHE": str(tmp_path)}
    for args in (("--help",), ("autotune", "--help"), ("profile", "show")):
        run = subprocess.run(
            [sys.executable, "-m", "vibeqc", *args],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert run.stdout


def test_generic_numerical_driver_supports_tuned_s_and_p_consumers():
    from tools.vibeqc_codegen.autotune import supported_schedule_trials
    from tools.vibeqc_codegen.benchmark import emit_shell_class_resource_cuda
    from tools.vibeqc_codegen.fused_schedule import build_fused_shell_plan
    from tools.vibeqc_codegen.ir import KernelConsumer
    from tools.vibeqc_codegen.shell_spec import FUSED_SHELL_SPEC_BY_NAME
    from tools.vibeqc_validation.f_shell_cuda import emit_numerical_driver

    for name in ("ssss", "pppp"):
        for consumer in ("force", "fock"):
            spec = FUSED_SHELL_SPEC_BY_NAME[name]
            trial = supported_schedule_trials(spec, consumer)[0]
            consumers = (
                (KernelConsumer.FOCK, KernelConsumer.FORCE)
                if consumer == "fock"
                else (KernelConsumer.FORCE,)
            )
            plan = build_fused_shell_plan(
                spec, consumers=consumers, schedule=trial.schedule
            )
            source = emit_shell_class_resource_cuda(spec, plan)
            driver = emit_numerical_driver(
                name, plan=plan, source=source, consumer=consumer
            )
            assert f'"uhf_{consumer}_persistent"' in driver


def test_native_source_identity_and_probe_abi_match_checkout():
    import ctypes

    from vibeqc import _native
    from vibeqc.autotune import source_identity

    library = _native.load_library()
    library.vibeqc_get_source_identity.restype = ctypes.c_char_p
    assert library.vibeqc_get_source_identity().decode() == source_identity(
        Path(__file__).resolve().parents[2]
    )
    descriptor = (
        profiles.DeviceDescriptor()
    )  # invalid size must fail before any GPU probe
    library.vibeqc_cuda_tuning_device.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(profiles.DeviceDescriptor),
    ]
    library.vibeqc_cuda_tuning_device.restype = ctypes.c_int
    assert library.vibeqc_cuda_tuning_device(0, ctypes.byref(descriptor)) != 0


def test_workload_without_direct_counters_is_a_noop_but_other_errors_propagate(
    monkeypatch,
):
    from types import SimpleNamespace

    from vibeqc import _autotune_worker

    error = [NotImplementedError("cached ERI route")]
    result = SimpleNamespace(
        items=[
            SimpleNamespace(
                iterations=1,
                energy=-1.0,
                forces=np.zeros((1, 3)),
                converged=True,
                executed_backend="cuda",
            )
        ]
    )

    class Prepared:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, **kwargs):
            return result

        def set_warm_start_updates(self, enabled):
            pass

        def last_shell_class_profile(self):
            raise error[0]

    calculator = SimpleNamespace(
        prepare_batch=lambda *a, **k: Prepared(), profile_diagnostics={}
    )
    monkeypatch.setattr(_autotune_worker, "Calculator", lambda **kwargs: calculator)
    workload = {
        "method": "rhf",
        "basis": "sto-3g",
        "device_id": 0,
        "representation": "cartesian",
        "atoms": [["He", [0.0, 0.0, 0.0]]],
        "batch": 1,
        "charge": 0,
        "multiplicity": 1,
    }
    measured = _autotune_worker.execute_workload(workload, profile=True)
    assert measured["work"] == [] and measured["profiling_reason"]
    assert measured["energies"] == [-1.0]
    error[0] = RuntimeError("CUDA execution failure")
    with pytest.raises(RuntimeError, match="CUDA execution failure"):
        _autotune_worker.execute_workload(workload, profile=True)
