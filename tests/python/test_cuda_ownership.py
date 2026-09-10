"""Prevent new CUDA files and stale semantic anchors from escaping the ledger."""

import pytest

from tools.report_cuda_ownership import code_lines, ownership_report


def ledger_for(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/sample.cu").write_text(
        "// header\nvoid runtime() {}\n// scientific section\nvoid formula() {}\n"
    )
    return {
        "schema": "vibeqc.cuda-ownership.v1",
        "subsystems": {
            "sample": {
                name: "explicit test policy"
                for name in (
                    "owner",
                    "current_default",
                    "generated_capability",
                    "missing_capability",
                    "retirement_condition",
                    "evidence",
                    "status",
                )
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
        "generated_families": [{"name": "test", "outputs": ["generated/*.cuh"]}],
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
