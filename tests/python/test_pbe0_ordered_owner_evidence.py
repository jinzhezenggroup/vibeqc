"""Retained owner-search evidence must reject corrupt semantics, not just bytes."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def evidence() -> tuple[ModuleType, dict[str, str]]:
    path = ROOT / "benchmarks/results/pbe0-ordered-ao-owners-20261003/verify.py"
    spec = importlib.util.spec_from_file_location("ordered_owner_evidence", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.load_bundle()


def test_complete_endpoints_keep_negative_changed_geometry_result(
    evidence: tuple[ModuleType, dict[str, str]],
) -> None:
    verifier, members = evidence
    summary = verifier.verify_all(members)
    assert summary == json.loads(members["receipts/owner-summary.json"])
    assert summary["native_endpoints_checked"] == 48
    largest = summary["cases"][-1]["variants"]
    assert largest["candidate"]["phase_iterations"]["moved"] == [13]
    assert largest["baseline"]["phase_iterations"]["moved"] == [12]
    assert (
        largest["candidate"]["phase_medians_seconds"]["moved"]
        > largest["baseline"]["phase_medians_seconds"]["moved"]
    )
    verifier.verify_basis(members)
    verifier.verify_qualification(members)


@pytest.mark.parametrize(
    "corruption",
    ["force", "repeat", "pair_work", "discovery", "budget", "source", "library", "job"],
)
def test_semantic_gates_ignore_claimed_pass_flags(
    evidence: tuple[ModuleType, dict[str, str]], corruption: str
) -> None:
    verifier, members = evidence
    prefix = "owner-endpoints/96/candidate/"
    record = json.loads(members[prefix + "local.json"])
    campaign = json.loads(members[prefix + "local.campaign.json"])
    call = campaign["force_calls"][0]
    if corruption == "force":
        record["records"][-1]["forces"][-1][-1] += 1e-4
    elif corruption == "repeat":
        record["records"].pop()
    elif corruption == "pair_work":
        call["grid_pair_visits"] -= 1
    elif corruption == "discovery":
        call["resident_ao_selection"]["work"]["discoveries"] = 0
    elif corruption == "budget":
        work = call["resident_ao_selection"]["work"]
        work["numeric_peak_bound_bytes"] = work["budget_bytes"] + 1
    elif corruption == "source":
        campaign["source_identity"] = "0" * 64
    elif corruption == "library":
        campaign["library_sha256"] = "0" * 64
    else:
        campaign["job"] = "5574"
    members[prefix + "local.json"] = json.dumps(record)
    members[prefix + "local.campaign.json"] = json.dumps(campaign)
    with pytest.raises(AssertionError):
        verifier.verify_all(members)


def test_actual_owner_labels_not_only_claimed_monotonicity(
    evidence: tuple[ModuleType, dict[str, str]],
) -> None:
    verifier, members = evidence
    basis = json.loads(members["receipts/owner-basis.json"])
    basis["cases"][-1]["owner_labels"].reverse()
    members["receipts/owner-basis.json"] = json.dumps(basis)
    with pytest.raises(AssertionError):
        verifier.verify_basis(members)


def test_missing_racecheck_completion_is_not_qualified(
    evidence: tuple[ModuleType, dict[str, str]],
) -> None:
    verifier, members = evidence
    name = "receipts/qualify-owner-kernels-n1.log"
    members[name] = members[name].replace("RACECHECK SUMMARY: 0 hazards", "incomplete")
    with pytest.raises(AssertionError):
        verifier.verify_qualification(members)
