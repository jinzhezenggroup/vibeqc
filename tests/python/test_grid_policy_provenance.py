"""Bind standard provenance to exact prescribed grid data, never just version."""

from dataclasses import replace

import pytest
from vibeqc._ks_snapshot import NativeKsSnapshot
from vibeqc_compiler.dft.grid import GridPolicy, grid_policy_provenance


@pytest.mark.parametrize("method", ["lda", "pbe"])
@pytest.mark.parametrize("accuracy", ["standard", "tight"])
def test_exact_policy_keeps_canonical_provenance(method: str, accuracy: str) -> None:
    policy = GridPolicy(accuracy)
    assert grid_policy_provenance(policy.resolve(method)) == policy.provenance


@pytest.mark.parametrize(
    "field",
    ["radial_points", "element_radii", "partition_iterations", "coincident_tolerance"],
)
def test_custom_grid_cannot_claim_upstream_policy(field: str) -> None:
    spec = GridPolicy().resolve("pbe")
    changes = {
        "radial_points": spec.radial_points + 1,
        "element_radii": ((1, 2.0),),
        "partition_iterations": 2,
        "coincident_tolerance": 1e-10,
    }
    provenance = grid_policy_provenance(replace(spec, **{field: changes[field]}))
    assert provenance == {"policy_version": 2, "contract": "explicit-grid-v2"}


def test_snapshot_has_write_once_grid_provenance_storage() -> None:
    snapshot = NativeKsSnapshot.__new__(NativeKsSnapshot)
    snapshot.grid_provenance = {"policy_version": 1, "contract": "reference-grid-v1"}
    with pytest.raises(AttributeError, match="immutable"):
        snapshot.grid_provenance = {"policy_version": 2}


def test_snapshot_owns_read_only_grid_provenance() -> None:
    original = grid_policy_provenance(GridPolicy().resolve("pbe"))
    snapshot = NativeKsSnapshot.__new__(NativeKsSnapshot)
    snapshot.grid_provenance = original
    original["radii_source"] = "modified-by-caller"
    assert snapshot.grid_provenance == GridPolicy().provenance
    with pytest.raises(TypeError):
        snapshot.grid_provenance["radii_source"] = "modified-through-snapshot"
    with pytest.raises(TypeError):
        del snapshot.grid_provenance["radii_source"]


def test_snapshot_preserves_absent_legacy_cuda_provenance() -> None:
    snapshot = NativeKsSnapshot.__new__(NativeKsSnapshot)
    snapshot.grid_provenance = None
    assert snapshot.grid_provenance is None
