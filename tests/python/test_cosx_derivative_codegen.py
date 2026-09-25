"""Compiler ownership checks for COSX derivative contractions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from vibeqc_compiler.method.cosx_derivative_runtime import (
    build_cosx_bidirectional_update_program,
    build_cosx_esp_derivative_update_program,
    build_cosx_molecular_ao_update_program,
    build_cosx_molecular_cotangent_program,
    build_cosx_pair_scale_program,
    build_cosx_point_gradient_update_program,
    build_cosx_projection_update_program,
    build_cosx_scale_program,
    build_cosx_symmetric_projection_update_program,
)
from vibeqc_compiler.tensor import execute
from vibeqc_compiler.tensor.program import Program

ROOT = Path(__file__).resolve().parents[2]


def _run(program: Program, **feeds: float) -> dict[str, float]:
    outputs = execute(
        program,
        {name: np.asarray(value, dtype=np.float64) for name, value in feeds.items()},
    ).outputs
    return {name: float(np.asarray(value)) for name, value in outputs.items()}


def test_cosx_scalar_contractions_match_retired_formulas() -> None:
    projection = _run(
        build_cosx_projection_update_program(),
        accumulator=0.7,
        left=-0.3,
        right=0.4,
    )
    np.testing.assert_allclose(
        projection["updated"], 0.7 + (-0.3) * 0.4, rtol=0, atol=0
    )

    esp = _run(
        build_cosx_esp_derivative_update_program(),
        accumulator=-0.2,
        matrix_derivative=0.3,
        projected_value=-0.4,
        matrix_value=0.5,
        projected_derivative=0.6,
    )
    np.testing.assert_allclose(
        esp["updated"], -0.2 + (0.3 * -0.4 + 0.5 * 0.6), rtol=0, atol=1e-16
    )

    symmetric = _run(
        build_cosx_symmetric_projection_update_program(),
        accumulator=0.11,
        ao=-0.7,
        density_rc=0.2,
        density_cr=0.4,
    )
    np.testing.assert_allclose(
        symmetric["updated"], 0.11 + (-0.7 * 0.5) * (0.2 + 0.4), rtol=0, atol=1e-16
    )

    bidirectional = _run(
        build_cosx_bidirectional_update_program(),
        right=0.1,
        left=-0.2,
        matrix_rc=0.3,
        matrix_cr=-0.4,
        projected=0.5,
        symmetric_projection=0.6,
    )
    np.testing.assert_allclose(
        bidirectional["right_updated"], 0.1 + 0.3 * 0.5, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        bidirectional["left_updated"], -0.2 + -0.4 * 0.6, rtol=0, atol=0
    )

    point = _run(
        build_cosx_point_gradient_update_program(),
        accumulator=0.09,
        density=-0.8,
        phi_derivative_row=0.11,
        potential_column=0.12,
        phi_row=0.13,
        potential_derivative_column=0.14,
        phi_derivative_column=0.15,
        potential_row=0.16,
        phi_column=0.17,
        potential_derivative_row=0.18,
    )
    raw_rc = 0.11 * 0.12 + 0.13 * 0.14
    raw_cr = 0.15 * 0.16 + 0.17 * 0.18
    expected_point = 0.09 + (-0.8 * 0.5) * (raw_rc + raw_cr)
    np.testing.assert_allclose(point["updated"], expected_point, rtol=0, atol=1e-16)

    molecular = _run(
        build_cosx_molecular_ao_update_program(),
        from_left=0.2,
        from_right=-0.1,
        density_rc=0.3,
        density_cr=0.5,
        potential=-0.7,
        left_potential=0.9,
    )
    np.testing.assert_allclose(
        molecular["from_left_updated"],
        0.2 + (0.5 * (0.3 + 0.5)) * -0.7,
        rtol=0,
        atol=1e-16,
    )
    np.testing.assert_allclose(
        molecular["from_right_updated"], -0.1 + 0.3 * 0.9, rtol=0, atol=1e-16
    )

    cotangent = _run(
        build_cosx_molecular_cotangent_program(),
        energy_factor=-0.25,
        weight=0.8,
        from_left=0.3,
        from_right=-0.2,
    )
    np.testing.assert_allclose(
        cotangent["cotangent"], (-0.25 * 0.8) * (0.3 - 0.2), rtol=0, atol=1e-16
    )

    scale = _run(build_cosx_scale_program(), factor=-0.25, value=0.8)
    pair_scale = _run(
        build_cosx_pair_scale_program(), first=-0.25, second=0.8, value=0.3
    )
    np.testing.assert_allclose(scale["scaled"], -0.25 * 0.8, rtol=0, atol=0)
    np.testing.assert_allclose(
        pair_scale["scaled"], (-0.25 * 0.8) * 0.3, rtol=0, atol=0
    )


def test_generated_header_retires_native_cosx_contraction_formulas(
    tmp_path: Path,
) -> None:
    output = tmp_path / "generated_cosx_derivative_contractions.cuh"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_cosx_derivative_native.py"),
            "--output",
            str(output),
        ],
        check=True,
        timeout=60,
    )
    generated = output.read_text()
    assert "#define VIBEQC_COSX_DERIVATIVE_HD __host__ __device__" in generated
    for helper in (
        "accumulate_projection",
        "accumulate_esp_derivative",
        "accumulate_symmetric_projection",
        "accumulate_bidirectional",
        "accumulate_point_gradient",
        "accumulate_molecular_ao",
        "molecular_cotangent",
        "scale_pair",
    ):
        assert helper in generated

    consumer = (ROOT / "src/dft/cuda_cosx_derivative.cu").read_text()
    assert '#include "generated_cosx_derivative_contractions.cuh"' in consumer
    for helper in (
        "accumulate_projection",
        "accumulate_esp_derivative",
        "accumulate_symmetric_projection",
        "accumulate_bidirectional",
        "accumulate_point_gradient",
        "accumulate_molecular_ao",
        "molecular_cotangent",
    ):
        assert f"generated_cosx_derivative::{helper}" in consumer

    for retired in (
        "value += ao[point * nbf + row] * density[row * nbf + column]",
        "value += matrix_derivative[row * nbf + column] * projected_value[column]",
        "const double raw_rc =",
        "0.5 * (density[orbital * nbf + column]",
        "scalar += symmetric_projection[point * nbf + row] * potential[point * nbf + row]",
    ):
        assert retired not in consumer

    cmake = (ROOT / "cmake/VibeQCGeneratedSources.cmake").read_text()
    assert "vibeqc_cosx_derivative_contraction_codegen" in cmake
    assert "generate_cosx_derivative_native.py" in cmake
