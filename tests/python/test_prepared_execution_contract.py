from types import SimpleNamespace

import pytest
from vibeqc_compiler.common.prepared_execution import (
    PreparedArtifactBinding,
    PreparedExecutionLease,
    PreparedExecutionMismatch,
    PreparedExecutionRequest,
)
from vibeqc_compiler.common.provenance import canonical_hash


def digest(value: object) -> str:
    return canonical_hash(value)


def request(**changes: object) -> PreparedExecutionRequest:
    values = {
        "kind": "test",
        "scientific_identity": digest("science"),
        "target_identity": digest("target"),
        "schedule_identity": digest("schedule"),
        "workspace_identity": digest("workspace"),
        "device": 0,
        "specialization_identity": digest("selection"),
        "capture_identity": digest("capture"),
    }
    values.update(changes)
    return PreparedExecutionRequest(**values)


def artifact(name: str = "a") -> PreparedArtifactBinding:
    return PreparedArtifactBinding(digest(("key", name)), digest(("binary", name)))


def test_prepared_lease_reuses_exact_identity_and_rechecks_budgets() -> None:
    lease = PreparedExecutionLease()
    req = request()
    contract = lease.install(req, (artifact(),), host_bytes=13, device_bytes=17)

    assert lease.require(req, max_host_bytes=13, max_device_bytes=17) is contract
    assert lease.identity == contract.identity
    with pytest.raises(ValueError, match="host budget"):
        lease.require(req, max_host_bytes=12, max_device_bytes=17)
    with pytest.raises(ValueError, match="device budget"):
        lease.require(req, max_host_bytes=13, max_device_bytes=16)


@pytest.mark.parametrize(
    "change",
    [
        {"scientific_identity": digest("science-2")},
        {"target_identity": digest("target-2")},
        {"schedule_identity": digest("schedule-2")},
        {"workspace_identity": digest("workspace-2")},
        {"specialization_identity": digest("selection-2")},
        {"capture_identity": digest("capture-2")},
        {"device": 1},
    ],
)
def test_prepared_lease_invalidates_all_compiled_region_identity_axes(
    change: dict[str, object],
) -> None:
    lease = PreparedExecutionLease()
    lease.install(request(), (artifact(),), host_bytes=1, device_bytes=2)
    with pytest.raises(PreparedExecutionMismatch, match="identity changed"):
        lease.require(request(**change), max_host_bytes=10, max_device_bytes=10)


def test_failure_requires_explicit_consumer_refresh_before_publication() -> None:
    lease = PreparedExecutionLease()
    lease.install(request(), (artifact(),), host_bytes=1, device_bytes=2)
    lease.mark_failure()
    assert lease.needs_refresh
    with pytest.raises(RuntimeError, match="requires refresh"):
        lease.mark_success()
    lease.mark_refresh()
    lease.mark_success()
    assert lease.executions == 1
    assert lease.refreshes == 1
    assert not lease.failed


def test_invalidation_is_explicit_and_artifact_binding_uses_existing_identity() -> None:
    binding = PreparedArtifactBinding.from_artifact(
        SimpleNamespace(
            metadata={"key": digest("artifact"), "binary_sha256": digest("binary")}
        )
    )
    lease = PreparedExecutionLease()
    lease.install(request(), (binding,), host_bytes=0, device_bytes=0)
    lease.invalidate()
    assert lease.contract is None
    assert lease.invalidations == 1
    with pytest.raises(RuntimeError, match="not installed"):
        lease.require(request(), max_host_bytes=0, max_device_bytes=0)
