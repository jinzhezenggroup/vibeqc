from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from check_periodic_asset_inventory import InventoryError, load_inventory, validate_inventory  # noqa: E402

INVENTORY = ROOT / "docs/periodic_asset_inventory.json"


def _fixture_repo(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    payload = json.loads(INVENTORY.read_text(encoding="utf-8"))
    for asset in payload["assets"]:
        path = tmp_path / asset["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generic periodic fixture\n", encoding="utf-8")
        for evidence in asset["evidence"]:
            evidence_path = tmp_path / evidence
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.touch()
    for evidence in payload["strain_derivative"]["evidence"]:
        path = tmp_path / evidence
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    notice = tmp_path / payload["provenance"]["notice"]
    notice.parent.mkdir(parents=True, exist_ok=True)
    notice.touch()
    return tmp_path, payload


def test_repository_periodic_asset_inventory_is_valid() -> None:
    validate_inventory(ROOT, load_inventory(INVENTORY))


def test_generic_periodic_owner_cannot_depend_on_gfn2(tmp_path: Path) -> None:
    root, payload = _fixture_repo(tmp_path)
    generic = next(
        asset for asset in payload["assets"] if asset["classification"] == "generic-lattice"
    )
    (root / generic["path"]).write_text("from model.gfn2.periodic import Cell\n", encoding="utf-8")

    with pytest.raises(InventoryError, match="depends on GFN2-specific internals"):
        validate_inventory(root, payload)


def test_retired_native_pbc_owner_cannot_reappear(tmp_path: Path) -> None:
    root, payload = _fixture_repo(tmp_path)
    retired = root / payload["retired_native_pbc_paths"][0]
    retired.parent.mkdir(parents=True, exist_ok=True)
    retired.touch()

    with pytest.raises(InventoryError, match="reappeared"):
        validate_inventory(root, payload)


def test_backend_coverage_vocabulary_is_fail_closed(tmp_path: Path) -> None:
    root, payload = _fixture_repo(tmp_path)
    broken = copy.deepcopy(payload)
    broken["assets"][0]["cuda"] = "probably-supported"

    with pytest.raises(InventoryError, match="unknown cuda state"):
        validate_inventory(root, broken)


def test_strain_derivative_cannot_be_promoted_without_inventory_update(tmp_path: Path) -> None:
    root, payload = _fixture_repo(tmp_path)
    broken = copy.deepcopy(payload)
    broken["strain_derivative"]["state"] = "implemented"

    with pytest.raises(InventoryError, match="strain_derivative.state"):
        validate_inventory(root, broken)


def test_historical_periodic_manifest_reappearance_requires_inventory_update(tmp_path: Path) -> None:
    root, payload = _fixture_repo(tmp_path)
    manifest = root / payload["provenance"]["historical_periodic_manifest"]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.touch()

    with pytest.raises(InventoryError, match="manifest reappeared"):
        validate_inventory(root, payload)
