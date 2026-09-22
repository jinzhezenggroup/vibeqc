"""Production cutover checks for compiler-owned GFN2 CUDA H0-force math."""

from __future__ import annotations

import subprocess
import sys
import typing
from pathlib import Path

import numpy as np
from vibeqc_compiler.method.gfn2_h0_force_runtime import (
    build_gfn2_h0_ao_update_program,
    build_gfn2_h0_distance_program,
    build_gfn2_h0_distance_vjp_program,
    build_gfn2_h0_offsite_factor_program,
    build_gfn2_h0_offsite_vjp_program,
    build_gfn2_h0_onsite_factor_program,
    build_gfn2_h0_onsite_vjp_program,
    build_gfn2_h0_pulay_seed_program,
)
from vibeqc_compiler.tensor import execute

ROOT = Path(__file__).resolve().parents[2]

if typing.TYPE_CHECKING:
    from vibeqc_compiler.tensor import Program


def _run(program: Program, feeds: dict[str, float]) -> dict[str, float]:
    live = {node.attrs["name"] for node in program.live_nodes if node.op == "input"}
    outputs = execute(
        program,
        {name: np.asarray(feeds[name], dtype=np.float64) for name in live},
    ).outputs
    return {name: float(np.asarray(value)) for name, value in outputs.items()}


def _pair_inputs() -> dict[str, float]:
    return {
        "first_shell_level": -0.43,
        "second_shell_level": -0.31,
        "first_cn_scale": 0.017,
        "second_cn_scale": -0.023,
        "first_cn": 2.4,
        "second_cn": 1.7,
        "first_radius": 1.42,
        "second_radius": 1.16,
        "first_polynomial": 0.21,
        "second_polynomial": -0.08,
        "pair_scale": 0.93,
        "distance": 2.31,
    }


def test_h0_onsite_factor_and_cn_vjp() -> None:
    feeds = _pair_inputs()
    expected = 0.5 * (
        feeds["first_shell_level"]
        - feeds["first_cn_scale"] * feeds["first_cn"]
        + feeds["second_shell_level"]
        - feeds["second_cn_scale"] * feeds["second_cn"]
    )
    factor = _run(build_gfn2_h0_onsite_factor_program(), feeds)["factor"]
    np.testing.assert_allclose(factor, expected, rtol=0, atol=2e-16)

    bar = -0.61
    actual = _run(
        build_gfn2_h0_onsite_vjp_program(),
        {**feeds, "bar_factor": bar},
    )
    np.testing.assert_allclose(
        actual["bar_first_cn"],
        -0.5 * feeds["first_cn_scale"] * bar,
        rtol=0,
        atol=2e-16,
    )
    np.testing.assert_allclose(
        actual["bar_second_cn"],
        -0.5 * feeds["second_cn_scale"] * bar,
        rtol=0,
        atol=2e-16,
    )


def test_h0_offsite_factor_and_generated_radial_vjp() -> None:
    feeds = _pair_inputs()
    average = 0.5 * (
        feeds["first_shell_level"]
        - feeds["first_cn_scale"] * feeds["first_cn"]
        + feeds["second_shell_level"]
        - feeds["second_cn_scale"] * feeds["second_cn"]
    )
    reduced = np.sqrt(
        feeds["distance"] / (feeds["first_radius"] + feeds["second_radius"])
    )
    first_shape = 1.0 + feeds["first_polynomial"] * reduced
    second_shape = 1.0 + feeds["second_polynomial"] * reduced
    spatial = feeds["pair_scale"] * first_shape * second_shape
    expected = average * spatial
    factor = _run(build_gfn2_h0_offsite_factor_program(), feeds)["factor"]
    np.testing.assert_allclose(factor, expected, rtol=0, atol=3e-16)

    bar = 0.73
    actual = _run(
        build_gfn2_h0_offsite_vjp_program(),
        {**feeds, "bar_factor": bar},
    )
    np.testing.assert_allclose(
        actual["bar_first_cn"],
        -0.5 * feeds["first_cn_scale"] * spatial * bar,
        rtol=0,
        atol=4e-16,
    )
    np.testing.assert_allclose(
        actual["bar_second_cn"],
        -0.5 * feeds["second_cn_scale"] * spatial * bar,
        rtol=0,
        atol=4e-16,
    )
    spatial_derivative = (
        feeds["pair_scale"]
        * (
            feeds["first_polynomial"] * second_shape
            + feeds["second_polynomial"] * first_shape
        )
        * reduced
        / (2.0 * feeds["distance"])
    )
    np.testing.assert_allclose(
        actual["bar_distance"],
        average * spatial_derivative * bar,
        rtol=2e-15,
        atol=4e-16,
    )


def test_h0_ao_contraction_is_compiler_owned() -> None:
    feeds = {
        "density": -0.37,
        "overlap": 0.29,
        "factor": -0.18,
        "overlap_adjoint": 0.07,
        "block_weight": -0.04,
    }
    actual = _run(build_gfn2_h0_ao_update_program(), feeds)
    np.testing.assert_allclose(
        actual["overlap_adjoint_updated"],
        feeds["overlap_adjoint"] + feeds["density"] * feeds["factor"],
        rtol=0,
        atol=2e-17,
    )
    np.testing.assert_allclose(
        actual["block_weight_updated"],
        feeds["block_weight"] + feeds["density"] * feeds["overlap"],
        rtol=0,
        atol=2e-17,
    )


def test_h0_distance_and_cartesian_vjp_are_generated() -> None:
    feeds = {"dx": 0.3, "dy": -0.4, "dz": 1.2}
    actual = _run(build_gfn2_h0_distance_program(), feeds)
    expected_squared = sum(value * value for value in feeds.values())
    expected_distance = np.sqrt(expected_squared)
    np.testing.assert_allclose(
        actual["distance_squared"], expected_squared, rtol=0, atol=2e-16
    )
    np.testing.assert_allclose(
        actual["distance"], expected_distance, rtol=0, atol=2e-16
    )

    bar = -0.57
    adjoint = _run(
        build_gfn2_h0_distance_vjp_program(),
        {**feeds, "bar_distance": bar},
    )
    for axis in ("x", "y", "z"):
        np.testing.assert_allclose(
            adjoint[f"bar_d{axis}"],
            bar * feeds[f"d{axis}"] / expected_distance,
            rtol=2e-15,
            atol=2e-16,
        )


def test_h0_pulay_seed_is_compiler_owned() -> None:
    actual = _run(
        build_gfn2_h0_pulay_seed_program(),
        {"seed": 0.41, "weighted": -0.17},
    )
    np.testing.assert_allclose(actual["pulay_seed"], 0.58, rtol=0, atol=1e-16)


def test_h0_offsite_preserves_native_left_to_right_product_range() -> None:
    feeds = _pair_inputs()
    feeds.update(
        {
            "first_cn_scale": 0.0,
            "second_cn_scale": 0.0,
            "first_radius": 0.5,
            "second_radius": 0.5,
            "distance": 1.0,
            "first_polynomial": 1.0e300,
            "second_polynomial": 1.0e100,
            "pair_scale": 1.0e-300,
        }
    )
    factor = _run(build_gfn2_h0_offsite_factor_program(), feeds)["factor"]
    assert np.isfinite(factor)


def test_generated_header_and_runtime_retire_handwritten_h0_force_math(
    tmp_path: Path,
) -> None:
    output = tmp_path / "generated_gfn2_h0_native.hpp"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_gfn2_h0_native.py"),
            "--output",
            str(output),
        ],
        check=True,
        timeout=60,
    )
    generated = output.read_text()
    assert "#define VIBEQC_GFN2_H0_HD __host__ __device__" in generated
    assert "VIBEQC_GFN2_H0_HD inline bool gfn2_h0_offsite_factor_tensor" in generated
    assert "VIBEQC_GFN2_H0_HD inline bool gfn2_h0_offsite_vjp_tensor" in generated
    assert "VIBEQC_GFN2_H0_HD inline bool gfn2_h0_ao_update_tensor" in generated
    assert "VIBEQC_GFN2_H0_HD inline bool gfn2_h0_distance_tensor" in generated
    assert "VIBEQC_GFN2_H0_HD inline bool gfn2_h0_distance_vjp_tensor" in generated
    assert "VIBEQC_GFN2_H0_HD inline bool gfn2_h0_pulay_seed_tensor" in generated
    assert "bar_distance" in generated

    consumer = (ROOT / "src/xtb/native/src/backends/cuda/gfn2_h0_force.cu").read_text()
    assert '#include "generated_gfn2_h0_native.hpp"' in consumer
    assert "evaluate_gfn2_h0_offsite_factor" in consumer
    assert "evaluate_gfn2_h0_offsite_vjp" in consumer
    assert "accumulate_gfn2_h0_ao" in consumer
    assert "evaluate_gfn2_h0_distance" in consumer
    assert "evaluate_gfn2_h0_distance_vjp" in consumer
    assert "evaluate_gfn2_h0_pulay_seed" in consumer
    for retired in (
        "spatial_scale_derivative",
        "polynomial_derivative",
        "level_weight",
        "radial_derivative",
        "overlap_contribution",
        "weight_contribution",
        "coordinate_scale",
        "const double pulay_seed = seed - weighted",
        "distance_squared = dx * dx",
    ):
        assert retired not in consumer

    cmake = (ROOT / "cmake/VibeQCGeneratedSources.cmake").read_text()
    assert "generate_gfn2_h0_native.py" in cmake
    assert "vibeqc_gfn2_h0_native_codegen" in cmake


def test_cpu_h0_and_cuda_values_share_the_generated_primal_and_ad() -> None:
    cpu = (ROOT / "src/xtb/native/src/model/gfn2/h0.cpp").read_text()
    cuda = (ROOT / "src/xtb/native/src/backends/cuda/gfn2_integrals.cu").read_text()
    for source in (cpu, cuda):
        assert '#include "generated_gfn2_h0_native.hpp"' in source
        assert "evaluate_gfn2_h0_offsite_factor" in source
        assert "evaluate_gfn2_h0_onsite_factor" in source
        assert "const double reduced_distance" not in source
    assert "evaluate_gfn2_h0_offsite_vjp" in cpu
    assert "evaluate_gfn2_h0_onsite_vjp" in cpu
    assert "evaluate_gfn2_h0_distance_vjp" in cpu
    assert "spatial_scale_derivative" not in cpu
    assert "polynomial_derivative" not in cpu
