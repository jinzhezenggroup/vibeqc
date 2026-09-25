"""Generate CPU/CUDA helpers for compiler-owned COSX derivative contractions."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import vibeqc_compiler

for package_name in ("tensor", "method"):
    qualified = f"vibeqc_compiler.{package_name}"
    if qualified not in sys.modules:
        package = types.ModuleType(qualified)
        package.__path__ = [str(ROOT / "python" / "vibeqc_compiler" / package_name)]
        package.__package__ = qualified
        sys.modules[qualified] = package
        setattr(vibeqc_compiler, package_name, package)

from vibeqc_compiler.method.cosx_derivative_runtime import (
    COSX_DERIVATIVE_RUNTIME_VERSION,
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
from vibeqc_compiler.tensor.program import Program
from vibeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp


def _device(source: str, function_name: str) -> str:
    needle = f"inline bool {function_name}("
    replacement = f"VIBEQC_COSX_DERIVATIVE_HD inline bool {function_name}("
    if source.count(needle) != 1:
        raise ValueError(f"expected one generated declaration for {function_name}")
    return source.replace(needle, replacement, 1)


def _emit(program: Program, name: str, inputs: tuple[str, ...], outputs: tuple[str, ...]) -> str:
    return _device(
        emit_scalar_cpp(
            program,
            function_name=name,
            input_order=inputs,
            output_order=outputs,
            check_intermediates=False,
        ),
        name,
    )


def native_header() -> str:
    projection = build_cosx_projection_update_program()
    esp_derivative = build_cosx_esp_derivative_update_program()
    symmetric = build_cosx_symmetric_projection_update_program()
    bidirectional = build_cosx_bidirectional_update_program()
    point_gradient = build_cosx_point_gradient_update_program()
    molecular_ao = build_cosx_molecular_ao_update_program()
    molecular_cotangent = build_cosx_molecular_cotangent_program()
    scale = build_cosx_scale_program()
    pair_scale = build_cosx_pair_scale_program()

    generated = "\n".join(
        (
            _emit(
                projection,
                "cosx_projection_update_tensor",
                ("accumulator", "left", "right"),
                ("updated",),
            ),
            _emit(
                esp_derivative,
                "cosx_esp_derivative_update_tensor",
                (
                    "accumulator",
                    "matrix_derivative",
                    "projected_value",
                    "matrix_value",
                    "projected_derivative",
                ),
                ("updated",),
            ),
            _emit(
                symmetric,
                "cosx_symmetric_projection_update_tensor",
                ("accumulator", "ao", "density_rc", "density_cr"),
                ("updated",),
            ),
            _emit(
                bidirectional,
                "cosx_bidirectional_update_tensor",
                (
                    "right",
                    "left",
                    "matrix_rc",
                    "matrix_cr",
                    "projected",
                    "symmetric_projection",
                ),
                ("right_updated", "left_updated"),
            ),
            _emit(
                point_gradient,
                "cosx_point_gradient_update_tensor",
                (
                    "accumulator",
                    "density",
                    "phi_derivative_row",
                    "potential_column",
                    "phi_row",
                    "potential_derivative_column",
                    "phi_derivative_column",
                    "potential_row",
                    "phi_column",
                    "potential_derivative_row",
                ),
                ("updated",),
            ),
            _emit(
                molecular_ao,
                "cosx_molecular_ao_update_tensor",
                (
                    "from_left",
                    "from_right",
                    "density_rc",
                    "density_cr",
                    "potential",
                    "left_potential",
                ),
                ("from_left_updated", "from_right_updated"),
            ),
            _emit(
                molecular_cotangent,
                "cosx_molecular_cotangent_tensor",
                ("energy_factor", "weight", "from_left", "from_right"),
                ("cotangent",),
            ),
            _emit(scale, "cosx_scale_tensor", ("factor", "value"), ("scaled",)),
            _emit(
                pair_scale,
                "cosx_pair_scale_tensor",
                ("first", "second", "value"),
                ("scaled",),
            ),
        )
    )

    hashes = "\n".join(
        f'inline constexpr const char* {name}_hash = "{program.logical_hash}";'
        for name, program in (
            ("projection", projection),
            ("esp_derivative", esp_derivative),
            ("symmetric_projection", symmetric),
            ("bidirectional", bidirectional),
            ("point_gradient", point_gradient),
            ("molecular_ao", molecular_ao),
            ("molecular_cotangent", molecular_cotangent),
            ("scale", scale),
            ("pair_scale", pair_scale),
        )
    )

    return f"""// Generated by tools/generate_cosx_derivative_native.py from TensorIR; do not edit.
#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define VIBEQC_COSX_DERIVATIVE_HD __host__ __device__
#else
#define VIBEQC_COSX_DERIVATIVE_HD
#endif

namespace vibeqc::dft::generated_cosx_derivative {{

inline constexpr const char* runtime_version = "{COSX_DERIVATIVE_RUNTIME_VERSION}";
{hashes}

{generated}

VIBEQC_COSX_DERIVATIVE_HD inline bool accumulate_projection(
    double left, double right, double& accumulator) noexcept {{
  double updated = 0.0;
  if (!cosx_projection_update_tensor(accumulator, left, right, updated)) return false;
  accumulator = updated;
  return true;
}}

VIBEQC_COSX_DERIVATIVE_HD inline bool accumulate_esp_derivative(
    double matrix_derivative, double projected_value, double matrix_value,
    double projected_derivative, double& accumulator) noexcept {{
  double updated = 0.0;
  if (!cosx_esp_derivative_update_tensor(
          accumulator, matrix_derivative, projected_value, matrix_value,
          projected_derivative, updated))
    return false;
  accumulator = updated;
  return true;
}}

VIBEQC_COSX_DERIVATIVE_HD inline bool accumulate_symmetric_projection(
    double ao, double density_rc, double density_cr, double& accumulator) noexcept {{
  double updated = 0.0;
  if (!cosx_symmetric_projection_update_tensor(
          accumulator, ao, density_rc, density_cr, updated))
    return false;
  accumulator = updated;
  return true;
}}

VIBEQC_COSX_DERIVATIVE_HD inline bool accumulate_bidirectional(
    double matrix_rc, double matrix_cr, double projected, double symmetric_projection,
    double& right, double& left) noexcept {{
  double right_updated = 0.0, left_updated = 0.0;
  if (!cosx_bidirectional_update_tensor(
          right, left, matrix_rc, matrix_cr, projected, symmetric_projection,
          right_updated, left_updated))
    return false;
  right = right_updated;
  left = left_updated;
  return true;
}}

VIBEQC_COSX_DERIVATIVE_HD inline bool accumulate_point_gradient(
    double density, double phi_derivative_row, double potential_column,
    double phi_row, double potential_derivative_column,
    double phi_derivative_column, double potential_row,
    double phi_column, double potential_derivative_row,
    double& accumulator) noexcept {{
  double updated = 0.0;
  if (!cosx_point_gradient_update_tensor(
          accumulator, density, phi_derivative_row, potential_column,
          phi_row, potential_derivative_column, phi_derivative_column,
          potential_row, phi_column, potential_derivative_row, updated))
    return false;
  accumulator = updated;
  return true;
}}

VIBEQC_COSX_DERIVATIVE_HD inline bool accumulate_molecular_ao(
    double density_rc, double density_cr, double potential, double left_potential,
    double& from_left, double& from_right) noexcept {{
  double from_left_updated = 0.0, from_right_updated = 0.0;
  if (!cosx_molecular_ao_update_tensor(
          from_left, from_right, density_rc, density_cr, potential, left_potential,
          from_left_updated, from_right_updated))
    return false;
  from_left = from_left_updated;
  from_right = from_right_updated;
  return true;
}}

VIBEQC_COSX_DERIVATIVE_HD inline bool molecular_cotangent(
    double energy_factor, double weight, double from_left, double from_right,
    double& cotangent) noexcept {{
  return cosx_molecular_cotangent_tensor(
      energy_factor, weight, from_left, from_right, cotangent);
}}

VIBEQC_COSX_DERIVATIVE_HD inline bool scale(
    double factor, double value, double& scaled) noexcept {{
  return cosx_scale_tensor(factor, value, scaled);
}}

VIBEQC_COSX_DERIVATIVE_HD inline bool scale_pair(
    double first, double second, double value, double& scaled) noexcept {{
  return cosx_pair_scale_tensor(first, second, value, scaled);
}}

}}  // namespace vibeqc::dft::generated_cosx_derivative

#undef VIBEQC_COSX_DERIVATIVE_HD
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(native_header(), encoding="utf-8")


if __name__ == "__main__":
    main()
