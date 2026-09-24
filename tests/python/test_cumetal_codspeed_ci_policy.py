"""Keep CuMetal PR performance telemetry bounded and correctness-gated."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_cumetal_codspeed_pr_tier_is_bounded_and_build_gated() -> None:
    workflow = (ROOT / ".github/workflows/cumetal-cuda.yml").read_text(encoding="utf-8")
    section = workflow.split("\n  cumetal-benchmark:\n", 1)[1].split(
        "\n  upload-test-results:\n", 1
    )[0]

    assert "needs: cuda-tests" in section
    assert "needs.cuda-tests.result == 'success'" in section
    assert (
        "VIBEQC_CUMETAL_CODSPEED_TIER: "
        "${{ github.event_name == 'pull_request' && 'pr' || 'full' }}"
    ) in section
    assert "benchmarks/cumetal_d4_codspeed.cu" in section
    assert "benchmarks/test_cumetal_d4_codspeed.py" in section

    harness = (ROOT / "benchmarks/test_cumetal_fp32_codspeed.py").read_text(
        encoding="utf-8"
    )
    assert 'return ("memory",)' in harness
    assert '_FULL_PROXY_WORKLOADS = ("memory", "gather", "mixed")' in harness

    d4 = (ROOT / "benchmarks/cumetal_d4_codspeed.cu").read_text(encoding="utf-8")
    validation = d4.index("validate_production_schedule();")
    ready = d4.index('std::cout << "READY device="')
    assert validation < ready
    assert '#include "dft/dispersion/d4_cuda.cu"' in d4
    assert "evaluate_d4_fixed_charge(" in d4
