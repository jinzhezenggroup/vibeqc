"""Exercise inventory guard boundaries without manufacturing capability evidence."""

from pathlib import Path
from typing import Any

import pytest

from tools import check_periodic_asset_inventory as inventory

FRAGMENTS = ("model/gfn2/", "gfn2_", "gfn2.")


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "repo"
    root.mkdir()
    assets = []
    for index, kind in enumerate(sorted(inventory.EXPECTED_CLASSIFICATIONS)):
        name = f"asset{index}.txt"
        (root / name).write_text("generic fixture only\n")
        assets.append(
            {
                "path": name,
                "classification": kind,
                "role": "test fixture only",
                "cpu": "source-present",
                "cuda": "not-applicable",
                "evidence": [name],
            }
        )
    (root / "notice.txt").write_text("fixture only\n")
    payload = {
        "schema_version": 1,
        "issue": 805,
        "classifications": sorted(inventory.EXPECTED_CLASSIFICATIONS),
        "backend_states": sorted(inventory.EXPECTED_BACKEND_STATES),
        "assets": assets,
        "retired_native_pbc_paths": sorted(inventory.EXPECTED_RETIRED_PATHS),
        "strain_derivative": {
            "state": "plumbing-only",
            "reason": "no science in fixture",
            "evidence": [assets[0]["path"]],
        },
        "generic_dependency_policy": {
            "forbidden_fragments": list(FRAGMENTS),
            "reason": "retain required dependency boundary",
        },
        "provenance": {
            "notice": "notice.txt",
            "current_native_sources_license": "test-only",
            "upstream_reference_project": "test-only",
            "upstream_reference_license": "test-only",
            "upstream_periodic_reference_revision": "a" * 40,
            "historical_periodic_manifest": "missing.json",
            "historical_periodic_manifest_state": "missing-current-tree",
        },
    }
    return root, payload


def _symlink(path: Path, target: Path) -> None:
    try:
        path.symlink_to(target)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks unavailable: {error}")


@pytest.mark.parametrize("field", ["asset", "evidence", "strain", "notice", "manifest"])
@pytest.mark.parametrize("style", ["absolute", "parent", "symlink"])
def test_inventory_paths_stay_in_repository(
    tmp_path: Path, field: str, style: str
) -> None:
    root, payload = _fixture(tmp_path)
    outside = tmp_path / "outside.txt"
    if field != "manifest":
        outside.write_text("external file must not count\n")
    raw = str(outside) if style == "absolute" else "../outside.txt"
    if style == "symlink":
        _symlink(root / "alias", outside)
        raw = "alias"
    if field == "asset":
        payload["assets"][0]["path"] = raw
    elif field == "evidence":
        payload["assets"][0]["evidence"] = [raw]
    elif field == "strain":
        payload["strain_derivative"]["evidence"] = [raw]
    elif field == "notice":
        payload["provenance"]["notice"] = raw
    else:
        payload["provenance"]["historical_periodic_manifest"] = raw
    with pytest.raises(inventory.InventoryError):
        inventory.validate_inventory(root, payload)


@pytest.mark.parametrize(
    "key,value", [("schema_version", True), ("schema_version", 1.0), ("issue", 805.0)]
)
def test_integer_identity_does_not_accept_json_coercions(
    tmp_path: Path, key: str, value: object
) -> None:
    root, payload = _fixture(tmp_path)
    payload[key] = value
    with pytest.raises(inventory.InventoryError, match=key):
        inventory.validate_inventory(root, payload)


@pytest.mark.parametrize("removed", FRAGMENTS)
def test_inventory_cannot_disable_its_dependency_guard(
    tmp_path: Path, removed: str
) -> None:
    root, payload = _fixture(tmp_path)
    generic = next(
        a for a in payload["assets"] if a["classification"] == "generic-lattice"
    )
    (root / generic["path"]).write_text(removed + "lattice\n")
    payload["generic_dependency_policy"]["forbidden_fragments"].remove(removed)
    with pytest.raises(inventory.InventoryError):
        inventory.validate_inventory(root, payload)


@pytest.mark.parametrize("where", ["inventory", "generic"])
def test_invalid_utf8_has_a_controlled_diagnostic(tmp_path: Path, where: str) -> None:
    root, payload = _fixture(tmp_path)
    if where == "inventory":
        path = root / "inventory.json"
        path.write_bytes(b"\xff")
        with pytest.raises(inventory.InventoryError):
            inventory.load_inventory(path)
    else:
        generic = next(
            a for a in payload["assets"] if a["classification"] == "generic-lattice"
        )
        (root / generic["path"]).write_bytes(b"\xff")
        with pytest.raises(inventory.InventoryError):
            inventory.validate_inventory(root, payload)


def test_valid_fixture_and_internal_alias_remain_valid(tmp_path: Path) -> None:
    root, payload = _fixture(tmp_path)
    inventory.validate_inventory(root, payload)
    _symlink(root / "alias", root / payload["assets"][0]["path"])
    payload["assets"][0]["evidence"] = ["alias"]
    inventory.validate_inventory(root, payload)


def test_dependency_guard_still_rejects_forbidden_source(tmp_path: Path) -> None:
    root, payload = _fixture(tmp_path)
    generic = next(
        a for a in payload["assets"] if a["classification"] == "generic-lattice"
    )
    (root / generic["path"]).write_text("from gfn2.lattice import Cell\n")
    with pytest.raises(inventory.InventoryError, match="depends on GFN2"):
        inventory.validate_inventory(root, payload)
