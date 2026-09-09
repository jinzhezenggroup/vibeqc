"""Synthetic legality/identity gates are independent of optional SDK hardware."""

from dataclasses import replace

import pytest

from tools.vibeqc_codegen.backend import TargetInfo, TargetScheduleShape
from tools.vibeqc_codegen.runtime_backend import (
    CompiledArtifactIdentity,
    ExecutionShape,
    LibraryRequest,
    RuntimeCapabilities,
    UnsupportedBackendFeature,
    UnsupportedLibraryProvider,
)


@pytest.mark.parametrize("subgroup", [8, 16, 32, 64])
def test_synthetic_subgroups_and_partial_final_workgroups(subgroup):
    target = RuntimeCapabilities("synthetic", True, 256, 32768, subgroup)
    shape = ExecutionShape(2 * subgroup, 128, subgroup)
    shape.validate_for(target)
    assert shape.padded_items(2 * subgroup + 1) == 4 * subgroup
    with pytest.raises(ValueError, match="subgroup"):
        replace(shape, subgroup_size=subgroup * 2).validate_for(target)
    with pytest.raises(ValueError, match="workspace"):
        replace(shape, local_bytes=32769).validate_for(target)


def test_unknown_subgroup_and_optional_graphs_do_not_preclude_scalar_execution():
    generic = TargetInfo("opencl", "queried device", None, 256, None)
    TargetScheduleShape(17, None).validate_for(generic)
    with pytest.raises(ValueError, match="subgroup"):
        TargetScheduleShape(32, 32).validate_for(generic)
    target = RuntimeCapabilities("opencl", True, 1024, 49152)
    ExecutionShape(17).validate_for(target)
    with pytest.raises(UnsupportedBackendFeature, match="not known"):
        ExecutionShape(32, subgroup_size=32).validate_for(target)
    for feature in (
        "requires_fp64_atomic_add",
        "requires_graphs",
        "requires_device_enqueue",
    ):
        with pytest.raises(UnsupportedBackendFeature):
            ExecutionShape(32, **{feature: True}).validate_for(target)
    with pytest.raises(UnsupportedBackendFeature, match="FP64"):
        ExecutionShape(32).validate_for(replace(target, fp64=False))


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_runtime_extents_fail_before_submission(value):
    target = RuntimeCapabilities("opencl", True, 1024, 0)
    with pytest.raises(ValueError):
        ExecutionShape(value).validate_for(target)


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_capabilities_and_requirements_reject_truthy_nonbooleans(value):
    with pytest.raises(ValueError, match="boolean capability"):
        RuntimeCapabilities("opencl", value, 256, 32768)
    target = RuntimeCapabilities("opencl", True, 256, 32768)
    with pytest.raises(ValueError, match="boolean requirement"):
        ExecutionShape(32, requires_graphs=value).validate_for(target)


def test_executable_identity_changes_without_changing_scientific_identity():
    identity = CompiledArtifactIdentity(
        "opencl",
        "uuid",
        "runtime",
        "compiler",
        "driver",
        "OpenCL C 1.2",
        ("-cl-std=CL1.2",),
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )
    for field, value in {
        "backend": "another",
        "device": "other-uuid",
        "runtime": "new-runtime",
        "compiler": "new-compiler",
        "driver": "new-driver",
        "language": "OpenCL C 3.0",
        "options": (),
        "source_hash": "d" * 64,
        "schedule_hash": "e" * 64,
    }.items():
        other = replace(identity, **{field: value})
        assert other.scientific_hash == identity.scientific_hash
        assert other.key != identity.key
        with pytest.raises(ValueError, match="incompatible"):
            identity.require_compatible(other)
    identity.require_compatible(replace(identity))


@pytest.mark.parametrize(
    "operation,shape",
    [("gemm", (3, 4, 5)), ("symmetric_eigh", (4,)), ("cholesky", (4,))],
)
def test_missing_vendor_library_never_reports_cpu_success(operation, shape):
    request = LibraryRequest(operation, shape, workspace_limit_bytes=8192)
    provider = UnsupportedLibraryProvider(
        "opencl", "no native library provider configured"
    )
    with pytest.raises(UnsupportedBackendFeature, match=operation):
        provider.plan(request)


def test_atomic_local_executable_cache_rejects_corruption_and_wrong_identity(
    tmp_path, monkeypatch
):
    import json

    from vibeqc import profiles

    from tools.vibeqc_codegen.artifact_cache import LocalArtifactCache

    cache = LocalArtifactCache(tmp_path / "private")
    identity = CompiledArtifactIdentity(
        "opencl",
        "uuid",
        "runtime",
        "compiler",
        "driver",
        "OpenCL C 1.2",
        (),
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )
    path = cache.install(identity, b"local compiler artifact")
    assert cache.load(identity) == b"local compiler artifact"
    original = path.read_bytes()

    def fail_replace(*_):
        raise OSError("simulated publication failure")

    with monkeypatch.context() as patch:
        patch.setattr(profiles.os, "replace", fail_replace)
        with pytest.raises(OSError, match="publication"):
            cache.install(identity, b"replacement")
    assert path.read_bytes() == original
    payload = json.loads(original)
    payload["identity"]["driver"] = "stale driver"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="incompatible"):
        cache.load(identity)
    payload = json.loads(original)
    payload["sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="corrupt"):
        cache.load(identity)
    for malformed in ([], {}, {"schema": "vibeqc.backend_artifact", "version": 1}):
        path.write_text(json.dumps(malformed))
        with pytest.raises(ValueError):
            cache.load(identity)


def test_cache_rejects_shared_write_directories_and_symlink_records(tmp_path):
    from tools.vibeqc_codegen.artifact_cache import LocalArtifactCache

    shared = tmp_path / "shared"
    shared.mkdir(mode=0o777)
    shared.chmod(0o777)
    with pytest.raises(ValueError, match="shared write"):
        LocalArtifactCache(shared)
    cache = LocalArtifactCache(tmp_path / "private")
    identity = CompiledArtifactIdentity(
        "opencl",
        "uuid",
        "runtime",
        "compiler",
        "driver",
        "OpenCL C 1.2",
        (),
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )
    (cache.directory / (identity.key + ".json")).symlink_to(
        tmp_path / "not-an-artifact"
    )
    with pytest.raises(OSError):
        cache.load(identity)
