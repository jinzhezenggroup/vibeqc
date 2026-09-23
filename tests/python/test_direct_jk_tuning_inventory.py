from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPOSITORY_ROOT / "tools/check_direct_jk_tuning_inventory.py"
SPEC = importlib.util.spec_from_file_location(
    "check_direct_jk_tuning_inventory", CHECKER_PATH
)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def _inventory() -> dict[str, object]:
    return json.loads(
        (REPOSITORY_ROOT / "docs/direct_jk_tuning_inventory.json").read_text(
            encoding="utf-8"
        )
    )


def _write_fixture(root: Path, payload: dict[str, object]) -> None:
    inventory = root / "docs/direct_jk_tuning_inventory.json"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    note = (
        root
        / ".agents/notes/implemented/performance/2026-09-21-direct-jk-target-resource-policy.md"
    )
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "# Direct J/K target-resource policy slice (#597)\n", encoding="utf-8"
    )

    constants = root / "src/scf/cuda/direct_constants.hpp"
    constants.parent.mkdir(parents=True, exist_ok=True)
    constants.write_text(
        "\n".join(
            [
                "constexpr std::size_t kPersistentEriAoLimit = 16;",
                "constexpr unsigned kSchwarzThreads = 1;",
                "constexpr std::size_t kBoundedGeneratedTasksPerShellPair = 1024;",
                "constexpr unsigned kPersistentForceAngularOrderCount = 7;",
                "constexpr unsigned kPersistentFockAngularOrderCount = 6;",
                "constexpr double kForceDensityProductScreeningTolerance = 1.0e-14;",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    policy = root / "src/scf/cuda/rhf_policy.hpp"
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text(
        "\n".join(
            [
                "fallback_persistent_eri_ao_limit{16}",
                "fallback_cublas_matrix_product_ao_threshold{17}",
                "SmallHfMatrixCalibration matrix{}",
                "resolve_small_hf_profitability",
                "maximum_task_capacity{8U * 1024U * 1024U}",
                "maximum_arena_bytes{std::size_t{1} << 30}",
                "cuda_stack_limit_bytes{std::size_t{64} << 10}",
                "maximum_persistent_quartet_warps_per_sm{8}",
                "resolve_direct_jk_schedule_policy",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    schedule = root / "python/vibeqc_compiler/integral/direct_resident_schedule.py"
    schedule.parent.mkdir(parents=True, exist_ok=True)
    schedule.write_text(
        "block_threads=128\nminimum_blocks_per_sm=4\nmaximum_bra_primitive_pairs=64\n",
        encoding="utf-8",
    )


def test_repository_inventory_is_valid() -> None:
    assert CHECKER.validate_repository(REPOSITORY_ROOT) == []


def test_checker_accepts_minimal_fixture(tmp_path: Path) -> None:
    payload = _inventory()
    _write_fixture(tmp_path, payload)
    assert CHECKER.validate_repository(tmp_path) == []


def test_checker_rejects_constant_drift(tmp_path: Path) -> None:
    payload = _inventory()
    _write_fixture(tmp_path, payload)
    constants = tmp_path / "src/scf/cuda/direct_constants.hpp"
    constants.write_text(
        constants.read_text(encoding="utf-8").replace("= 16;", "= 24;", 1),
        encoding="utf-8",
    )
    assert any(
        "kPersistentEriAoLimit" in error and "drifted" in error
        for error in CHECKER.validate_repository(tmp_path)
    )


def test_checker_rejects_missing_migrated_owner_token(tmp_path: Path) -> None:
    payload = _inventory()
    _write_fixture(tmp_path, payload)
    policy = tmp_path / "src/scf/cuda/rhf_policy.hpp"
    policy.write_text(
        policy.read_text(encoding="utf-8").replace(
            "maximum_persistent_quartet_warps_per_sm{8}\n", ""
        ),
        encoding="utf-8",
    )
    assert any(
        "persistent-quartet-workers-per-sm" in error and "required token" in error
        for error in CHECKER.validate_repository(tmp_path)
    )


def test_checker_rejects_unsafe_source_path(tmp_path: Path) -> None:
    payload = _inventory()
    remaining = payload["remaining_constants"]
    assert isinstance(remaining, list) and remaining
    modified = copy.deepcopy(payload)
    modified_remaining = modified["remaining_constants"]
    assert isinstance(modified_remaining, list)
    assert isinstance(modified_remaining[0], dict)
    modified_remaining[0]["source"] = "../outside.hpp"
    _write_fixture(tmp_path, modified)
    assert any(
        "must stay inside the repository" in error
        for error in CHECKER.validate_repository(tmp_path)
    )


def test_checker_rejects_unknown_classification(tmp_path: Path) -> None:
    payload = copy.deepcopy(_inventory())
    remaining = payload["remaining_constants"]
    assert isinstance(remaining, list) and isinstance(remaining[0], dict)
    remaining[0]["classification"] = "magic"
    _write_fixture(tmp_path, payload)
    assert any(
        "classification is not recognized" in error
        for error in CHECKER.validate_repository(tmp_path)
    )


def test_checker_rejects_missing_retirement_condition(tmp_path: Path) -> None:
    payload = copy.deepcopy(_inventory())
    remaining = payload["remaining_constants"]
    assert isinstance(remaining, list) and isinstance(remaining[0], dict)
    remaining[0]["retirement_condition"] = ""
    _write_fixture(tmp_path, payload)
    assert any(
        "retirement_condition must be non-empty" in error
        for error in CHECKER.validate_repository(tmp_path)
    )


def test_checker_rejects_duplicate_symbol(tmp_path: Path) -> None:
    payload = copy.deepcopy(_inventory())
    excluded = payload["excluded_from_tuning"]
    assert isinstance(excluded, list) and isinstance(excluded[0], dict)
    excluded[0]["symbol"] = "kPersistentEriAoLimit"
    _write_fixture(tmp_path, payload)
    assert any(
        "duplicate constant symbol" in error
        for error in CHECKER.validate_repository(tmp_path)
    )


@pytest.mark.parametrize("field", ["schema_version", "issue"])
def test_checker_requires_integer_identity(tmp_path: Path, field: str) -> None:
    payload = copy.deepcopy(_inventory())
    payload[field] = str(payload[field])
    _write_fixture(tmp_path, payload)
    assert any(field in error for error in CHECKER.validate_repository(tmp_path))
