"""Policy guards for bounded CPU RKS AO-grid reuse."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RKS = (ROOT / "src/dft/rks.cpp").read_text(encoding="utf-8")
XC = (ROOT / "src/dft/xc.cpp").read_text(encoding="utf-8")


def test_cpu_rks_ao_cache_is_bounded_and_has_streaming_fallback() -> None:
    assert "kCpuRksAoCacheMaximumBytes = 64ULL * 1024ULL * 1024ULL" in RKS
    assert "cache_bytes <= kCpuRksAoCacheMaximumBytes" in RKS
    assert "ao_cache.emplace(dft::prepare_rks_ao_cache" in RKS
    assert "if (!cache)" in XC
    assert "basis.evaluate(points.data() + 3 * begin" in XC


def test_cpu_rks_ao_cache_is_counted_as_retained_memory() -> None:
    assert "ao_cache ? ao_cache->numeric_capacity_bytes() : 0" in RKS
    assert "numeric_capacity_bytes() const noexcept" in XC


def test_cpu_rks_ao_cache_does_not_overlap_incremental_xc() -> None:
    assert "if (!incremental_xc && evaluate_xc.cached_direct)" in RKS


def test_pbe_rks_binds_cached_and_streamed_evaluators() -> None:
    binding = "RksXcEvaluator(evaluate_pbe_xc_rks, evaluate_pbe_xc_rks_cached)"
    assert RKS.count(binding) >= 3
    assert "integrate_pbe_rks_with_tail_scaled_cached" in XC
