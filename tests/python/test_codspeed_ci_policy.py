"""Guard the bounded PR CodSpeed tier and full performance coverage."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_codspeed_pr_tier_stays_bounded() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = workflow.split("\n  cpu-benchmark:\n", 1)[1].split(
        "\n  upload-coverage:\n", 1
    )[0]
    assert "if: github.event_name != 'schedule'" in job
    assert (
        "VIBEQC_CODSPEED_TIER: "
        "${{ github.event_name == 'pull_request' && 'pr' || 'full' }}"
    ) in job
    assert "cpu-benchmark" not in workflow.split("\n  pass:\n", 1)[1]

    benchmark = (ROOT / "benchmarks/test_cpu_codspeed.py").read_text(encoding="utf-8")
    assert benchmark.count("pr_fast=True") == 2
    assert '"water-rhf-sto3g"' in benchmark
    assert '"water-pbe-sto3g"' in benchmark
    assert '"water-wb97mv-smallgrid-sto3g"' in benchmark
    assert "grid_shape=(12, 4, 8)" in benchmark
