"""Prevent new CUDA files and stale semantic anchors from escaping the ledger."""

import copy
import json
from pathlib import Path

import pytest

from tools.compare_cuda_ownership import compare
from tools.report_cuda_ownership import code_lines, ownership_report, validate_baseline


def test_versioned_current_report_matches_maintained_source_and_ledger():
    """Keep the current source snapshot reproducible without materializing CUDA."""
    root = Path(__file__).resolve().parents[2]
    ledger = json.loads((root / "docs/cuda_ownership.json").read_text())
    current = json.loads((root / "docs/cuda_ownership_current.json").read_text())
    fresh = ownership_report(root, ledger)
    validate_baseline(current)
    for key in (
        "files",
        "maintained_code_lines",
        "all_handwritten_scientific_lines",
        "per_subsystem",
        "migration_ledger",
    ):
        assert current[key] == fresh[key], (
            "regenerate docs/cuda_ownership_current.json for the current source/ledger"
        )


def ledger_for(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/sample.cu").write_text(
        "// header\nvoid runtime() {}\n// scientific section\nvoid formula() {}\n"
    )
    return {
        "schema": "vibeqc.cuda-ownership.v1",
        "subsystems": {
            "sample": {
                "evidence": ["src/sample.cu"],
                **{
                    name: "explicit test policy"
                    for name in (
                        "owner",
                        "current_default",
                        "generated_capability",
                        "missing_capability",
                        "retirement_condition",
                        "status",
                    )
                },
            }
        },
        "files": [
            {
                "path": "src/sample.cu",
                "role": "runtime",
                "subsystem": "sample",
                "reason": "allocation/launch owner",
                "regions": [
                    {
                        "start": "void formula()",
                        "role": "scientific",
                        "reason": "mathematical equation",
                    }
                ],
            }
        ],
        "generated_families": [
            {"name": "test", "owner": "src/sample.cu", "outputs": ["generated/*.cuh"]}
        ],
    }


def test_complete_inventory_and_stale_region_fail_closed(tmp_path):
    ledger = ledger_for(tmp_path)
    report = ownership_report(tmp_path, ledger)
    assert report["maintained_code_lines"]["scientific"] == 1
    assert report["maintained_code_lines"]["runtime"] == 1
    (tmp_path / "src/new.cu").write_text("__global__ void unnoticed() {}\n")
    with pytest.raises(ValueError, match="unclassified"):
        ownership_report(tmp_path, ledger)
    (tmp_path / "src/new.cu").unlink()
    ledger["files"][0]["regions"][0]["start"] = "deleted formula name"
    with pytest.raises(ValueError, match="exactly once"):
        ownership_report(tmp_path, ledger)


def test_oracle_reclassification_cannot_claim_code_deletion(tmp_path):
    ledger = ledger_for(tmp_path)
    before = ownership_report(tmp_path, ledger)
    ledger["files"][0]["regions"][0]["role"] = "oracle"
    after = ownership_report(tmp_path, ledger)
    assert (
        after["all_handwritten_scientific_lines"]
        == before["all_handwritten_scientific_lines"]
    )
    assert after["maintained_code_lines"]["oracle"] == 1


def test_generated_build_output_is_separate_and_not_counted_twice(tmp_path):
    ledger = ledger_for(tmp_path)
    generated = tmp_path / "build/generated"
    generated.mkdir(parents=True)
    (generated / "kernel.cuh").write_text("// generated\nvoid kernel() {}\n")
    ledger["generated_families"][0]["outputs"].append("generated/kernel.cuh")
    report = ownership_report(tmp_path, ledger, tmp_path / "build")
    assert report["maintained_code_lines"]["scientific"] == 1
    assert len(report["generated"][0]["files"]) == 1
    assert report["generated"][0]["files"][0]["code_lines"] == 1
    assert report["generated_bytes"] > 0
    ledger["generated_families"].append(
        {**ledger["generated_families"][0], "name": "collision"}
    )
    with pytest.raises(ValueError, match="multiple families"):
        ownership_report(tmp_path, ledger, tmp_path / "build")


def test_stale_generated_owner_and_evidence_paths_are_rejected(tmp_path):
    ledger = ledger_for(tmp_path)
    ledger["generated_families"][0]["owner"] = "missing_generator.py"
    with pytest.raises(ValueError, match="stale owner"):
        ownership_report(tmp_path, ledger)
    ledger["generated_families"][0]["owner"] = "src/sample.cu"
    ledger["subsystems"]["sample"]["evidence"] = ["missing_evidence.json"]
    with pytest.raises(ValueError, match="stale evidence"):
        ownership_report(tmp_path, ledger)


def test_edited_baseline_aggregate_cannot_claim_retirement(tmp_path):
    report = ownership_report(tmp_path, ledger_for(tmp_path))
    validate_baseline(report)
    report["maintained_code_lines"]["scientific"] += 10
    with pytest.raises(ValueError, match="totals differ"):
        validate_baseline(report)


def test_physical_count_preserves_literals_and_drops_comments():
    text = '// only comment\nconst char* url = "https://example"; // comment\n/* spanning\ncomment */ int x = 1;\n\n'
    assert code_lines(text) == [False, True, False, True, False]
    assert code_lines("int x = 0xA'B'C; // ignored\n// only comment\n") == [True, False]
    assert code_lines(
        'const char* r = R"tag(// literal\n/* literal */)tag";\n// comment\n'
    ) == [True, True, False]


def test_overlapping_regions_and_new_cuda_header_are_rejected(tmp_path):
    ledger = ledger_for(tmp_path)
    (tmp_path / "src/new.hpp").write_text("__device__ void helper() {}\n")
    with pytest.raises(ValueError, match="unclassified"):
        ownership_report(tmp_path, ledger)
    (tmp_path / "src/new.hpp").unlink()
    ledger["files"][0]["regions"].append(
        {"start": "void runtime()", "role": "oracle", "reason": "overlapping claim"}
    )
    with pytest.raises(ValueError, match="overlapping"):
        ownership_report(tmp_path, ledger)


def test_physical_edits_and_unchanged_reclassification_are_separate(tmp_path):
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    old_ledger = ledger_for(old_root)
    new_ledger = ledger_for(new_root)
    new_ledger["files"][0]["regions"][0]["role"] = "runtime"
    with (new_root / "src/sample.cu").open("a") as stream:
        stream.write("void another_runtime() {}\n")
    result = compare(
        old_root,
        ownership_report(old_root, old_ledger),
        new_root,
        ownership_report(new_root, new_ledger),
    )
    assert result["added"] == {"runtime": 1}
    assert result["removed"] == {}
    assert result["unchanged_lines_reclassified"] == {"scientific -> runtime": 1}
    assert result["net_role_delta"]["runtime"] == 2
    assert result["net_role_delta"]["scientific"] == -1


def test_stale_source_or_inconsistent_totals_cannot_claim_retirement(tmp_path):
    ledger = ledger_for(tmp_path)
    report = ownership_report(tmp_path, ledger)
    corrupted = copy.deepcopy(report)
    corrupted["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        compare(tmp_path, report, tmp_path, corrupted)
    corrupted = copy.deepcopy(report)
    corrupted["maintained_code_lines"]["scientific"] += 1
    with pytest.raises(ValueError, match="totals differ"):
        compare(tmp_path, report, tmp_path, corrupted)
