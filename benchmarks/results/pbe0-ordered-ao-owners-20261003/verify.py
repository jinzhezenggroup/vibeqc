"""Recompute ordered-owner endpoint gates, work equality and retained timings."""

import hashlib
import importlib.util
import json
import lzma
import math
from pathlib import Path
from types import ModuleType

import numpy as np

DIRECTORY = Path(__file__).resolve().parent
BASE = "a718695de66d04af272addb61b2b28ebd6700eb4"
IDENTITIES = {
    "baseline": (
        "46852852006dc81b164796e994bc335c4f336da3f03e38ba76bec3c3e0269111",
        "c2f7c6e192a83ff09816af2e0834c3a22fb0190009e7592a9f2e5c3025b04254",
    ),
    "candidate": (
        "0beb3d81d31dfefe257b6baa4c8b5d2d21d3b2df1e96ef37f11915d17a6207f3",
        "c071235850716c36fc0b1f682a8237967a00bf4ef715d8e1e19eb66177c1f462",
    ),
}


def endpoint_checker() -> ModuleType:
    """Reuse scientific/work gates, not the other campaign's claimed identities."""
    path = DIRECTORY.parent / "pbe0-resident-active-ao-20261003/verify.py"
    spec = importlib.util.spec_from_file_location("owner_endpoint_checks", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_bundle() -> dict[str, str]:
    """Check all exact UTF-8 member bytes before checking their semantics."""
    storage = json.loads((DIRECTORY / "storage.json").read_text())
    assert storage["schema"] == "generativeqc.lossless-utf8-evidence.v1"
    assert storage["measured_base_commit"] == BASE
    packed = (DIRECTORY / "campaign.json.xz").read_bytes()
    assert len(packed) == storage["stored_bytes"]
    assert hashlib.sha256(packed).hexdigest() == storage["sha256"]
    raw = lzma.decompress(packed)
    assert len(raw) == storage["uncompressed_bytes"]
    assert hashlib.sha256(raw).hexdigest() == storage["uncompressed_sha256"]
    members = json.loads(raw)
    assert len(members) == 40 and set(members) == set(storage["members"])
    for name, value in members.items():
        assert hashlib.sha256(value.encode()).hexdigest() == storage["members"][name]
    assert (
        hashlib.sha256(members["receipts/run-ao-endpoint.py"].encode()).hexdigest()
        == endpoint_checker().HARNESS
    )
    return members


def verify_all(members: dict[str, str]) -> dict:
    """Check every same-geometry reference repeat; never select only the best."""
    verifier = endpoint_checker()
    summary = {
        "scope": "Same GPU allocation, separate matching source/library pairs. Only the geometry owner-range search changes. Both variants use local force AO maps, dense SCF, 256-point tiles and original default budgets.",
        "reference_scope": "Reused independent numerical references, not fresh reference timings.",
        "cold_scope": "Ordered processes share compiler caches; cold compilation and iteration differences do not isolate the optimization.",
        "cases": [],
    }
    for atoms in (24, 96):
        variants = {}
        references = set()
        for variant, (source, library) in IDENTITIES.items():
            prefix = f"owner-endpoints/{atoms}/{variant}/"
            verifier._MEMBERS = {
                name.removeprefix(prefix): value
                for name, value in members.items()
                if name.startswith(prefix)
            }
            verifier.SOURCE, verifier.LIBRARY = source, library
            verifier.BASE, verifier.JOBS = BASE, ("5581", "5582", "not-used")
            reference_path = Path("reference.json")
            reference = verifier.load(reference_path)
            verifier.complete_rows(reference, atoms)
            references.add(verifier.digest(reference_path))
            result = verifier.verify_variant(
                Path("."), atoms, "local", reference, verifier.digest(reference_path)
            )
            rows = verifier.load(Path("local.json"))["records"]
            result["phase_ranges_seconds"] = {
                phase: [
                    min(
                        row["complete_seconds"] for row in rows if row["phase"] == phase
                    ),
                    max(
                        row["complete_seconds"] for row in rows if row["phase"] == phase
                    ),
                ]
                for phase in verifier.PHASES
            }
            variants[variant] = result
        assert len(references) == 1
        baseline, candidate = variants["baseline"], variants["candidate"]
        for field in ("job", "cuda_visible_devices", "grid_work_plan", "resources"):
            assert baseline[field] == candidate[field], (atoms, field)
        for old_call, new_call in zip(
            baseline["force_calls"], candidate["force_calls"], strict=True
        ):
            old_work = old_call["resident_ao_selection"]["work"]
            new_work = new_call["resident_ao_selection"]["work"]
            assert old_work.keys() == new_work.keys()
            assert all(
                old_work[key] == new_work[key]
                for key in old_work
                if key != "discovery_seconds"
            )
        reduction = (
            1
            - candidate["phase_medians_seconds"]["warm"]
            / baseline["phase_medians_seconds"]["warm"]
        )
        assert math.isfinite(reduction)
        summary["cases"].append(
            {
                "atoms": atoms,
                "variants": variants,
                "reference_sha256": references.pop(),
                "warm_reduction": reduction,
            }
        )
    summary["native_endpoints_checked"] = 48
    return summary


def verify_basis(members: dict[str, str]) -> None:
    """Establish ordered native ownership without mistaking it for GPU timing."""
    basis = json.loads(members["receipts/owner-basis.json"])
    assert basis["source_identity"] == IDENTITIES["candidate"][0]
    assert basis["job"] == "5592"
    assert [case["atoms"] for case in basis["cases"]] == [24, 96]
    for case in basis["cases"]:
        labels = np.asarray(case["owner_labels"], dtype="<i8")
        assert labels.shape == (case["aos"],) == (8 * case["atoms"],)
        assert labels.min() == 0 and labels.max() == case["atoms"] - 1
        assert case["nondecreasing"] and (labels[:-1] <= labels[1:]).all()
        assert hashlib.sha256(labels.tobytes()).hexdigest() == case["sha256"]


def verify_qualification(members: dict[str, str]) -> None:
    """Keep failed attempts distinct from completed routing and sanitizer gates."""
    failed = members["receipts/qualify-owner-n1.log"]
    assert "job=5579 host=node1" in failed and "5 failed, 14 passed" in failed
    current = members["receipts/qualify-owner-kernels-n1.log"]
    assert "job=5588 host=node1" in current
    assert "SOURCE IDENTITY PASS " + IDENTITIES["candidate"][0] in current
    assert current.count("6 passed, 54 deselected") == 5
    assert "8 passed, 18 deselected" in current
    assert current.count("ERROR SUMMARY: 0 errors") == 3
    assert "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" in current
    aot = members["receipts/qualify-owner-aot-n1.log"]
    assert "job=5589" in aot and "4 passed, 56 deselected" in aot
    scheduler = members["receipts/qualification-scheduler.txt"]
    assert "JobId=5588 " in scheduler and "JobState=COMPLETED" in scheduler
    assert "ExitCode=0:0" in scheduler and "TimeLimit=01:30:00" in scheduler
    assert "Partition=main" in scheduler and "gres/gpu:5090:1" in scheduler


def main() -> None:
    """Recompute the summary instead of trusting stored pass flags or medians."""
    members = load_bundle()
    summary = verify_all(members)
    assert summary == json.loads(members["receipts/owner-summary.json"])
    verify_basis(members)
    verify_qualification(members)
    print("PASS: 48 complete endpoints, all-reference gates, equal AO work/budgets")
    for case in summary["cases"]:
        print(case["atoms"], "warm reduction", case["warm_reduction"])


if __name__ == "__main__":
    main()
