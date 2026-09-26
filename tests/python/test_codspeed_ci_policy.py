"""Guard bounded, change-aware PR CodSpeed coverage and the full tier."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_codspeed_pr_tier_stays_bounded_and_change_aware() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = workflow.split("\n  cpu-benchmark:\n", 1)[1].split(
        "\n  upload-coverage:\n", 1
    )[0]
    assert "if: github.event_name != 'schedule'" in job
    assert (
        "VIBEQC_CODSPEED_TIER: "
        "${{ github.event_name == 'pull_request' && 'pr' || 'full' }}"
    ) in job
    assert "Select change-aware PR CodSpeed coverage" in job
    assert "VIBEQC_CODSPEED_EXTRA_CASES=" in job
    assert "src/dft/" in job
    assert "python/vibeqc_compiler/(dft|xc)/" in job
    assert "cpu-benchmark" not in workflow.split("\n  pass:\n", 1)[1]

    benchmark = (ROOT / "benchmarks/test_cpu_codspeed.py").read_text(encoding="utf-8")
    assert benchmark.count("pr_fast=True") == 2
    assert '"water-rhf-sto3g"' in benchmark
    assert '"water-pbe-sto3g"' in benchmark
    assert '"water-wb97mv-smallgrid-sto3g"' in benchmark
    assert 'pr_extra="wb97mv"' in benchmark
    assert "VIBEQC_CODSPEED_EXTRA_CASES" in benchmark
    assert "grid_shape=(12, 4, 8)" in benchmark
