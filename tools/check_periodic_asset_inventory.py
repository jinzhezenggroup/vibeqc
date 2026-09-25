from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

EXPECTED_CLASSIFICATIONS = {
    "generic-lattice",
    "gfn2-science",
    "temporary-bootstrap",
    "reusable-runtime",
}
EXPECTED_BACKEND_STATES = {
    "source-present",
    "contract-only",
    "plumbing-only",
    "backend-neutral",
    "unsupported",
    "not-applicable",
}
EXPECTED_RETIRED_PATHS = {
    "src/xtb/native/src/model/gfn2/periodic_topology.cpp",
    "src/xtb/native/src/model/gfn2/periodic_topology.hpp",
    "src/xtb/native/src/model/gfn2/periodic_integrals.cpp",
    "src/xtb/native/src/model/gfn2/periodic_integrals.hpp",
    "src/xtb/native/src/model/gfn2/periodic_ewald.cpp",
    "src/xtb/native/src/model/gfn2/periodic_ewald.hpp",
    "src/xtb/native/src/model/gfn2/periodic_multipole.cpp",
    "src/xtb/native/src/model/gfn2/periodic_multipole.hpp",
    "src/xtb/native/src/model/gfn2/lattice.cpp",
    "src/xtb/native/src/model/gfn2/lattice.hpp",
}
REQUIRED_FORBIDDEN_FRAGMENTS = {"model/gfn2/", "gfn2_", "gfn2."}
HEX40 = re.compile(r"^[0-9a-f]{40}$")


class InventoryError(ValueError):
    pass


def _require_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InventoryError(f"{where} must be a non-empty string")
    return value


def _require_string_list(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise InventoryError(f"{where} must be a non-empty list")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_require_string(item, f"{where}[{index}]"))
    return result


def load_inventory(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InventoryError(f"cannot load periodic asset inventory: {exc}") from exc
    if not isinstance(payload, dict):
        raise InventoryError("periodic asset inventory root must be an object")
    return payload


def _local_path(root: Path, raw: str) -> Path:
    """Keep retained and intentionally missing paths inside the canonical root."""
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise InventoryError(f"unsafe inventory path: {raw!r}")
    try:
        candidate = (root / relative).resolve()
        candidate.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise InventoryError(f"invalid repository-local path: {raw!r}") from exc
    return candidate


def validate_inventory(root: Path, payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise InventoryError("periodic asset inventory root must be an object")
    try:
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("root is not a directory")
    except (OSError, RuntimeError, ValueError) as exc:
        raise InventoryError("repository root must be an existing directory") from exc
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise InventoryError("schema_version must be 1")
    if type(payload.get("issue")) is not int or payload["issue"] != 805:
        raise InventoryError("issue must be 805")

    declared_classifications = set(
        _require_string_list(payload.get("classifications"), "classifications")
    )
    if declared_classifications != EXPECTED_CLASSIFICATIONS:
        raise InventoryError(
            "classifications must contain the canonical four-way ownership split"
        )

    declared_states = set(
        _require_string_list(payload.get("backend_states"), "backend_states")
    )
    if declared_states != EXPECTED_BACKEND_STATES:
        raise InventoryError(
            "backend_states must contain the canonical coverage vocabulary"
        )

    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise InventoryError("assets must be a non-empty list")

    paths: set[str] = set()
    seen_classifications: set[str] = set()
    generic_paths: list[Path] = []
    for index, asset in enumerate(assets):
        where = f"assets[{index}]"
        if not isinstance(asset, dict):
            raise InventoryError(f"{where} must be an object")
        path = _require_string(asset.get("path"), f"{where}.path")
        if path in paths:
            raise InventoryError(f"duplicate asset path: {path}")
        paths.add(path)
        actual_path = _local_path(root, path)
        if not actual_path.is_file():
            raise InventoryError(f"retained periodic asset is missing: {path}")

        classification = _require_string(
            asset.get("classification"), f"{where}.classification"
        )
        if classification not in EXPECTED_CLASSIFICATIONS:
            raise InventoryError(f"unknown classification for {path}: {classification}")
        seen_classifications.add(classification)
        if classification == "generic-lattice":
            generic_paths.append(actual_path)

        _require_string(asset.get("role"), f"{where}.role")
        for backend in ("cpu", "cuda"):
            state = _require_string(asset.get(backend), f"{where}.{backend}")
            if state not in EXPECTED_BACKEND_STATES:
                raise InventoryError(f"unknown {backend} state for {path}: {state}")

        for evidence in _require_string_list(
            asset.get("evidence"), f"{where}.evidence"
        ):
            if not _local_path(root, evidence).is_file():
                raise InventoryError(f"periodic evidence path is missing: {evidence}")

    missing_classifications = EXPECTED_CLASSIFICATIONS - seen_classifications
    if missing_classifications:
        joined = ", ".join(sorted(missing_classifications))
        raise InventoryError(
            f"periodic ownership inventory is missing classifications: {joined}"
        )

    retired_paths = set(
        _require_string_list(
            payload.get("retired_native_pbc_paths"), "retired_native_pbc_paths"
        )
    )
    if retired_paths != EXPECTED_RETIRED_PATHS:
        raise InventoryError(
            "retired_native_pbc_paths must retain the canonical retired owner set"
        )
    for retired in sorted(retired_paths):
        if _local_path(root, retired).exists() or (root / retired).is_symlink():
            raise InventoryError(
                f"retired native-PBC owner reappeared without inventory update: {retired}"
            )

    derivative = payload.get("strain_derivative")
    if not isinstance(derivative, dict):
        raise InventoryError("strain_derivative must be an object")
    if derivative.get("state") != "plumbing-only":
        raise InventoryError(
            "strain_derivative.state must remain explicit and fail closed"
        )
    _require_string(derivative.get("reason"), "strain_derivative.reason")
    for evidence in _require_string_list(
        derivative.get("evidence"), "strain_derivative.evidence"
    ):
        if not _local_path(root, evidence).is_file():
            raise InventoryError(
                f"strain-derivative evidence path is missing: {evidence}"
            )

    policy = payload.get("generic_dependency_policy")
    if not isinstance(policy, dict):
        raise InventoryError("generic_dependency_policy must be an object")
    fragments = _require_string_list(
        policy.get("forbidden_fragments"),
        "generic_dependency_policy.forbidden_fragments",
    )
    _require_string(policy.get("reason"), "generic_dependency_policy.reason")
    if not REQUIRED_FORBIDDEN_FRAGMENTS <= {fragment.lower() for fragment in fragments}:
        raise InventoryError(
            "forbidden_fragments must retain the GFN2 dependency guard"
        )
    for generic_path in generic_paths:
        try:
            text = generic_path.read_text(encoding="utf-8").lower()
        except (OSError, UnicodeError) as exc:
            raise InventoryError(
                f"cannot read generic periodic owner: {generic_path}"
            ) from exc
        for fragment in fragments:
            if fragment.lower() in text:
                relative = generic_path.relative_to(root)
                raise InventoryError(
                    f"generic periodic owner depends on GFN2-specific internals: {relative} contains {fragment!r}"
                )

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise InventoryError("provenance must be an object")
    notice = _require_string(provenance.get("notice"), "provenance.notice")
    if not _local_path(root, notice).is_file():
        raise InventoryError(f"periodic provenance notice is missing: {notice}")
    _require_string(
        provenance.get("current_native_sources_license"),
        "provenance.current_native_sources_license",
    )
    _require_string(
        provenance.get("upstream_reference_project"),
        "provenance.upstream_reference_project",
    )
    _require_string(
        provenance.get("upstream_reference_license"),
        "provenance.upstream_reference_license",
    )
    revision = _require_string(
        provenance.get("upstream_periodic_reference_revision"),
        "provenance.upstream_periodic_reference_revision",
    )
    if HEX40.fullmatch(revision) is None:
        raise InventoryError(
            "upstream_periodic_reference_revision must be a lowercase 40-hex commit"
        )
    manifest = _require_string(
        provenance.get("historical_periodic_manifest"),
        "provenance.historical_periodic_manifest",
    )
    if provenance.get("historical_periodic_manifest_state") != "missing-current-tree":
        raise InventoryError(
            "historical_periodic_manifest_state must explicitly describe the current tree"
        )
    if _local_path(root, manifest).exists() or (root / manifest).is_symlink():
        raise InventoryError(
            "historical periodic manifest reappeared; update its inventory state before claiming coverage"
        )


def check_repository(root: Path, inventory_path: Path | None = None) -> None:
    path = (
        inventory_path or root / "manifests/maintenance/periodic_asset_inventory.json"
    )
    validate_inventory(root, load_inventory(path))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate retained periodic/PBC asset ownership evidence"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    try:
        check_repository(args.root, args.inventory)
    except InventoryError as exc:
        print(f"periodic asset inventory check failed: {exc}")
        return 1
    print("periodic asset inventory check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
