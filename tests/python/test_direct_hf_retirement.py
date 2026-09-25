"""Guards for the issue-356 Direct-HF retirement classification."""

import copy
import json
from pathlib import Path

import pytest

from tools.check_direct_hf_retirement import (
    DEFAULT_LEDGER,
    validate_retirement_ledger,
)
from tools.report_cuda_ownership import load_ledger

ROOT = Path(__file__).resolve().parents[2]


def _inputs() -> tuple[dict, dict]:
    return (
        json.loads(DEFAULT_LEDGER.read_text()),
        load_ledger(ROOT / "docs/cuda_ownership"),
    )


def test_direct_hf_retirement_ledger_covers_every_scientific_direct_file_once() -> None:
    retirement, ownership = _inputs()
    summary = validate_retirement_ledger(ROOT, retirement, ownership)
    assert summary["issue"] == 356
    assert summary["scientific_files"] > 0
    assert summary["statuses"]["performance-exception"] > 0
    assert summary["statuses"]["unsupported-fallback"] > 0
    assert summary["statuses"]["scientific-policy"] > 0


def test_new_scientific_direct_file_must_be_classified() -> None:
    retirement, ownership = _inputs()
    ownership = copy.deepcopy(ownership)
    ownership["files"].append(
        {
            "path": "src/scf/cuda/direct_unclassified.cuh",
            "role": "scientific",
            "subsystem": "direct_integrals",
            "reason": "test-only unclassified science",
            "regions": [],
        }
    )
    with pytest.raises(ValueError, match="unclassified=.*direct_unclassified"):
        validate_retirement_ledger(ROOT, retirement, ownership)


def test_duplicate_retirement_family_ownership_is_rejected() -> None:
    retirement, ownership = _inputs()
    retirement = copy.deepcopy(retirement)
    duplicate = retirement["families"][0]["files"][0]
    retirement["families"][1]["files"].append(duplicate)
    with pytest.raises(ValueError, match="appears in both"):
        validate_retirement_ledger(ROOT, retirement, ownership)


def test_runtime_only_file_cannot_be_counted_as_retirement_science() -> None:
    retirement, ownership = _inputs()
    retirement = copy.deepcopy(retirement)
    runtime_path = next(
        row["path"]
        for row in ownership["files"]
        if row["path"].startswith("src/scf/cuda/direct_") and row["role"] == "runtime"
    )
    retirement["families"][0]["files"].append(runtime_path)
    with pytest.raises(ValueError, match="runtime-only"):
        validate_retirement_ledger(ROOT, retirement, ownership)


def test_scientific_family_cannot_claim_independent_oracle_status() -> None:
    retirement, ownership = _inputs()
    retirement = copy.deepcopy(retirement)
    retirement["families"][0]["status"] = "oracle"
    with pytest.raises(ValueError, match="oracle.*semantic"):
        validate_retirement_ledger(ROOT, retirement, ownership)


@pytest.mark.parametrize(
    "name",
    [
        "direct_force_low_order.cuh",
        "direct_force_order2.cuh",
        "direct_force_order3.cuh",
        "direct_native_psss.cuh",
    ],
)
def test_low_order_force_geometry_remains_scientific(name: str) -> None:
    """Generated scalar roots do not retire handwritten Gaussian geometry."""
    retirement, ownership = _inputs()
    path = f"src/scf/cuda/{name}"
    roles = {row["path"]: row["role"] for row in ownership["files"]}
    assert roles[path] == "scientific"
    owners = [family["id"] for family in retirement["families"] if path in family["files"]]
    assert owners == ["native-low-order-force"]


def test_generated_low_order_force_roots_remain_in_use() -> None:
    """Preserve generated force roots while tracking their native inputs honestly."""
    assert not (ROOT / "src/scf/cuda/direct_native_pair_order2_gradient.cuh").exists()
    low_order = (ROOT / "src/scf/cuda/direct_force_low_order.cuh").read_text(
        encoding="utf-8"
    )
    assert "generated_weighted_eri::ssss_force" in low_order
    psss = (ROOT / "src/scf/cuda/direct_native_psss.cuh").read_text(encoding="utf-8")
    assert "generated_weighted_eri::psss_force" in psss

    order2 = (ROOT / "src/scf/cuda/direct_force_order2.cuh").read_text(encoding="utf-8")
    for name in ("psps", "ppss", "dsss"):
        assert f"generated_weighted_eri::{name}_force" in order2

    order3 = (ROOT / "src/scf/cuda/direct_force_order3.cuh").read_text(encoding="utf-8")
    for name in ("ppps", "dsps", "dpss", "fsss"):
        assert f"generated_weighted_eri::{name}_force" in order3
