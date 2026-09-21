"""Retirement guards for compiler-owned Direct-HF high-order pair gradients."""

from pathlib import Path

from vibeqc_compiler.integral.direct_pair_gradient_cuda import (
    emit_direct_high_order_pair_gradient_header,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_high_order_pair_gradient_header_is_compiler_owned() -> None:
    source = emit_direct_high_order_pair_gradient_header()
    assert source.startswith("#pragma once\n")
    assert (
        "Generated from the compiler-owned subset/Wick pair-gradient lowering."
        in source
    )
    assert "static_assert(PairOrder <= 6);" in source
    assert "first_pair | second_pair | third_pair, 3U" in source
    assert "term.first_center[geometry.axes[differentiated]] += derivative;" in source


def test_order456_consumes_generated_pair_gradient_without_native_duplicate() -> None:
    order456 = (
        REPOSITORY_ROOT / "src/scf/cuda/direct_native_order456_gradient.cuh"
    ).read_text(encoding="utf-8")
    assert '#include "generated_direct_high_order_pair_gradient.cuh"' in order456
    assert "direct_native_pair_high_order_gradient.cuh" not in order456
    assert not (
        REPOSITORY_ROOT / "src/scf/cuda/direct_native_pair_high_order_gradient.cuh"
    ).exists()

    generated = (REPOSITORY_ROOT / "cmake/VibeQCGeneratedSources.cmake").read_text(
        encoding="utf-8"
    )
    cuda = (REPOSITORY_ROOT / "cmake/VibeQCCuda.cmake").read_text(encoding="utf-8")
    assert "VIBEQC_DIRECT_HIGH_ORDER_PAIR_GRADIENT_HEADER" in generated
    assert "generate_direct_pair_gradient.py" in generated
    assert cuda.count("VIBEQC_DIRECT_HIGH_ORDER_PAIR_GRADIENT_HEADER") == 2
