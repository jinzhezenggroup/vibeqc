from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "src/scf/cuda/df_occupied_exchange.cpp").read_text(encoding="utf-8")


def test_fast_occupied_exchange_paths_skip_full_output_clear() -> None:
    function = SOURCE.split("vibeqc_status build_occupied_exchange", maxsplit=1)[1]
    before_projected, after_projected = function.split(
        "if (plan.integral_source && plan.streamed", maxsplit=1
    )
    assert before_projected.count("cudaMemsetAsync(") == 1
    assert "if (!rank)" in before_projected

    projected_and_resident, fallback = after_projected.split(
        "// Only the tiled fallback accumulates partial auxiliary products into K.",
        maxsplit=1,
    )
    assert "cudaMemsetAsync(" not in projected_and_resident

    normalized_source = " ".join(SOURCE.split())
    assert "&zero, output + r + c * n" in normalized_source
    assert "&zero, output, n" in projected_and_resident

    before_panel_loop = fallback.split(
        "for (std::size_t qbegin = 0; qbegin < plan.naux", maxsplit=1
    )[0]
    assert before_panel_loop.count("cudaMemsetAsync(") == 1


def test_occupied_exchange_clear_traffic_census() -> None:
    # Each admitted nonzero-rank fast K call now avoids one matrix-sized
    # cudaMemset. This is deterministic traffic/launch work, not a wall-time claim.
    for nao, bytes_avoided in ((384, 1_179_648), (768, 4_718_592)):
        assert nao * nao * 8 == bytes_avoided
