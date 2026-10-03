"""Regression coverage for the #1598 default-promotion inventory."""

from __future__ import annotations

import copy
import shutil
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from tools.check_default_promotion_inventory import (
    DEFAULT_INVENTORY,
    ROOT,
    load_and_validate,
    validate_inventory,
)


def _payload() -> dict:
    payload, errors = load_and_validate(DEFAULT_INVENTORY, root=ROOT)
    assert not errors
    return payload


def _copy_audited_sources(payload: dict, destination_root: Path) -> None:
    sources = set(payload["scope"]["audited_sources"])
    for entry in payload["entries"]:
        sources.update(entry["sources"])
    for relative in sorted(sources):
        source = ROOT / relative
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def test_current_default_promotion_inventory_is_complete() -> None:
    payload, errors = load_and_validate(DEFAULT_INVENTORY, root=ROOT)
    assert not errors
    assert payload["tracking_issue"] == 1598
    assert len(payload["entries"]) >= 10


def test_fixture_copies_registered_sources_outside_the_audited_scope(
    tmp_path: Path,
) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    for entry in payload["entries"]:
        for relative in entry["sources"]:
            assert (tmp_path / relative).read_bytes() == (ROOT / relative).read_bytes()
    assert not validate_inventory(payload, root=tmp_path)


def test_inventory_requires_owner_rationale_and_revisit_condition() -> None:
    payload = _payload()
    for field, value in (
        ("owner_issues", []),
        ("rationale", ""),
        ("revisit_condition", ""),
    ):
        candidate = copy.deepcopy(payload)
        candidate["entries"][0][field] = value
        errors = validate_inventory(candidate, root=ROOT, check_sources=False)
        assert any(field in error for error in errors)


def test_inventory_rejects_duplicate_control_registration() -> None:
    payload = _payload()
    candidate = copy.deepcopy(payload)
    control = candidate["entries"][0]["controls"][0]
    candidate["entries"][1]["controls"].append(control)
    errors = validate_inventory(candidate, root=ROOT, check_sources=False)
    assert any("registered by both" in error for error in errors)


def test_inventory_rejects_missing_audited_control() -> None:
    payload = _payload()
    candidate = copy.deepcopy(payload)
    target = "tensor-execution:cuda-graph"
    for entry in candidate["entries"]:
        if target in entry["controls"]:
            entry["controls"].remove(target)
            break
    errors = validate_inventory(candidate, root=ROOT)
    assert any(target in error and "unregistered" in error for error in errors)


def test_new_tensor_schedule_opt_in_must_be_registered(tmp_path: Path) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)

    plan = tmp_path / "python/generativeqc_compiler/tensor/cuda_plan.py"
    source = plan.read_text()
    source = source.replace(
        "    direct_gemm: bool = True\n",
        "    new_default_off_path: bool = False\n    direct_gemm: bool = True\n",
        1,
    )
    assert "new_default_off_path" in source
    plan.write_text(source)

    errors = validate_inventory(payload, root=tmp_path)
    assert any(
        "tensor-schedule:new_default_off_path" in error and "unregistered" in error
        for error in errors
    )


@pytest.mark.parametrize("prefix", ["experimental_", "incremental_"])
@pytest.mark.parametrize(
    "initializer", ["{}", "{false}", " = false", "{true}", " = true", " = policy()", ""]
)
def test_new_scf_control_must_be_registered_independently_of_initializer(
    tmp_path: Path, prefix: str, initializer: str
) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    name = f"{prefix}new_path"
    source = tmp_path / "src/scf/types.hpp"
    original = source.read_text()
    changed = original.replace(
        "struct ScfOptions {", f"struct ScfOptions {{\n  bool {name}{initializer};", 1
    )
    assert changed != original
    source.write_text(changed)

    errors = validate_inventory(payload, root=tmp_path)
    assert any(
        f"scf-option:{name}" in error and "unregistered" in error for error in errors
    )


def test_commented_scf_declarations_are_not_controls(tmp_path: Path) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / "src/scf/types.hpp"
    source.write_text(
        source.read_text()
        + "\n// bool experimental_removed_path{false};\n"
        + "/* bool incremental_removed_path = false; */\n"
    )
    assert not validate_inventory(payload, root=tmp_path)


def test_scf_comment_separates_type_and_control_name(tmp_path: Path) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / "src/scf/types.hpp"
    source.write_text(
        source.read_text().replace(
            "struct ScfOptions {",
            "struct ScfOptions {\n  bool/* explanation */experimental_new_path{false};",
            1,
        )
    )
    errors = validate_inventory(payload, root=tmp_path)
    assert any("scf-option:experimental_new_path" in error for error in errors)


@pytest.mark.parametrize("initializer", ["{false}", " = false", " = policy()"])
def test_registered_scf_control_survives_initializer_changes(
    tmp_path: Path, initializer: str
) -> None:
    payload = _payload()
    _copy_audited_sources(payload, tmp_path)
    source = tmp_path / "src/scf/types.hpp"
    source.write_text(
        source.read_text().replace(
            "bool incremental_direct_jk{};",
            f"bool incremental_direct_jk{initializer};",
            1,
        )
    )
    assert not validate_inventory(payload, root=tmp_path)


def test_tensor_schedule_classifications_preserve_existing_decisions() -> None:
    entries = {
        control: entry
        for entry in _payload()["entries"]
        for control in entry["controls"]
    }
    for name in ("stream_reductions", "streamed_gemm_reduction"):
        entry = entries[f"tensor-schedule:{name}"]
        assert entry["classification"] == "negative-evidence"
        assert any("generated-reduction-h100" in item for item in entry["evidence"])
    assert entries["tensor-schedule:direct_gemm"]["classification"] == "already-default"
    assert (
        entries["tensor-schedule:layouts"]["classification"]
        == "guarded-promotion-candidate"
    )
