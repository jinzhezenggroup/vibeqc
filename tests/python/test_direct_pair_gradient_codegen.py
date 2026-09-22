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
        "Generated from the compiler-owned high-order primitive-gradient lowering."
        in source
    )
    assert "static_assert(PairOrder <= 6);" in source
    assert "first_pair | second_pair | third_pair, 3U" in source
    assert "term.first_center[geometry.axes[differentiated]] += derivative;" in source
    assert "__device__ inline void primitive_eri_order456_gradient(" in source
    assert "HighOrderCoulombWorkspace" in source
    assert "axis_wick_multiplicity" in source
    assert "high_order_coulomb<CoulombOrder>" in source
    assert "direct_native_high_order_coulomb.cuh" not in source
    assert "gradient[3][coordinate] =" in source


def test_order456_consumer_is_fully_compiler_owned() -> None:
    source = emit_direct_high_order_pair_gradient_header()
    assert "contracted_eri_cartesian_source_order4_gradient(" in source
    assert "contracted_eri_cartesian_source_order5_gradient(" in source
    assert "contracted_eri_cartesian_source_order6_gradient(" in source

    native = REPOSITORY_ROOT / "src/scf/cuda/direct_native_order456_gradient.cuh"
    assert not native.exists()
    native_coulomb = (
        REPOSITORY_ROOT / "src/scf/cuda/direct_native_high_order_coulomb.cuh"
    )
    assert not native_coulomb.exists()

    force = (REPOSITORY_ROOT / "src/scf/cuda/direct_force_quartet.cuh").read_text(
        encoding="utf-8"
    )
    assert '#include "generated_direct_high_order_pair_gradient.cuh"' in force
    assert "direct_native_order456_gradient.cuh" not in force

    generated = (REPOSITORY_ROOT / "cmake/VibeQCGeneratedSources.cmake").read_text(
        encoding="utf-8"
    )
    cuda = (REPOSITORY_ROOT / "cmake/VibeQCCuda.cmake").read_text(encoding="utf-8")
    assert "VIBEQC_DIRECT_HIGH_ORDER_PAIR_GRADIENT_HEADER" in generated
    assert "generate_direct_pair_gradient.py" in generated
    assert "coulomb_recurrence_cuda.py" in generated
    assert cuda.count("VIBEQC_DIRECT_HIGH_ORDER_PAIR_GRADIENT_HEADER") == 2
