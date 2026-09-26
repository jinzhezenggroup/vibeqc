"""Generate compiler-owned native GFN2 AES2 scalar kernels and reverse response."""

from __future__ import annotations

import argparse
import sys
import types
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "python"))

# Build-time generation must remain independent of NumPy/site packages.
import vibeqc_compiler

for package_name in ("tensor", "method"):
    qualified = f"vibeqc_compiler.{package_name}"
    if qualified not in sys.modules:
        package = types.ModuleType(qualified)
        package.__path__ = [str(ROOT / "python" / "vibeqc_compiler" / package_name)]
        package.__package__ = qualified
        sys.modules[qualified] = package
        setattr(vibeqc_compiler, package_name, package)

from vibeqc_compiler.method.gfn2_aes2 import (
    DIPOLE_NAMES,
    FIRST_DIPOLE_NAMES,
    FIRST_QUADRUPOLE_NAMES,
    GFN2_AES2_MULTIPOLE_KEXP,
    GFN2_AES2_MULTIPOLE_SHIFT,
    GFN2_AES2_RUNTIME_VERSION,
    QUADRUPOLE_NAMES,
    SECOND_DIPOLE_NAMES,
    SECOND_QUADRUPOLE_NAMES,
    build_gfn2_aes2_kernel_program,
    build_gfn2_aes2_kernel_vjp_program,
    build_gfn2_aes2_onsite_energy_program,
    build_gfn2_aes2_onsite_potential_program,
    build_gfn2_aes2_pair_energy_program,
    build_gfn2_aes2_pair_geometry_vjp_program,
    build_gfn2_aes2_pair_potential_program,
    build_gfn2_aes2_pair_vjp_compose_program,
    build_gfn2_aes2_radius_from_fraction_program,
)
from vibeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp


def _inputs(program: typing.Any) -> tuple[str, ...]:
    names = {node.attrs["name"] for node in program.live_nodes if node.op == "input"}
    return tuple(sorted(names))


def _emit(
    program: typing.Any,
    *,
    function_name: str,
    outputs: tuple[str, ...],
    cuda: bool,
    check_intermediates: bool = True,
) -> tuple[str, tuple[str, ...]]:
    inputs = _inputs(program)
    source = emit_scalar_cpp(
        program,
        function_name=function_name,
        input_order=inputs,
        output_order=outputs,
        check_intermediates=check_intermediates,
    )
    if cuda:
        # Match the established GFN2 CUDA scalar lowering: NVCC provides device
        # overloads for the std:: math calls emitted by scalar_cpp.
        source = source.replace("inline bool ", "__device__ inline bool ")
    return source, inputs


def _call(
    name: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    mapping: dict[str, str],
    output_mapping: dict[str, str],
) -> str:
    args = [mapping[input_name] for input_name in inputs]
    args += [output_mapping[output_name] for output_name in outputs]
    return f"{name}({', '.join(args)})"


def _header(*, cuda: bool) -> str:
    radius = build_gfn2_aes2_radius_from_fraction_program()
    kernels = build_gfn2_aes2_kernel_program()
    kernel_vjp = build_gfn2_aes2_kernel_vjp_program()
    onsite = build_gfn2_aes2_onsite_energy_program()
    onsite_vjp = build_gfn2_aes2_onsite_potential_program()
    pair = build_gfn2_aes2_pair_energy_program()
    pair_potential = build_gfn2_aes2_pair_potential_program()
    pair_geometry = build_gfn2_aes2_pair_geometry_vjp_program()
    pair_compose = build_gfn2_aes2_pair_vjp_compose_program()

    radius_body, radius_inputs = _emit(
        radius,
        function_name="gfn2_aes2_radius_from_fraction_tensor",
        outputs=("radius", "cn_derivative"),
        cuda=cuda,
    )
    kernel_body, kernel_inputs = _emit(
        kernels,
        function_name="gfn2_aes2_pair_kernels_tensor",
        outputs=("kernel3", "kernel5"),
        cuda=cuda,
        check_intermediates=False,
    )
    kernel_vjp_outputs = tuple(kernel_vjp.program.outputs)
    kernel_vjp_body, kernel_vjp_inputs = _emit(
        kernel_vjp.program,
        function_name="gfn2_aes2_pair_kernel_vjp_tensor",
        outputs=kernel_vjp_outputs,
        cuda=cuda,
        check_intermediates=False,
    )
    onsite_body, onsite_inputs = _emit(
        onsite,
        function_name="gfn2_aes2_onsite_energy_tensor",
        outputs=("energy",),
        cuda=cuda,
    )
    onsite_vjp_outputs = tuple(onsite_vjp.program.outputs)
    onsite_vjp_body, onsite_vjp_inputs = _emit(
        onsite_vjp.program,
        function_name="gfn2_aes2_onsite_potential_tensor",
        outputs=onsite_vjp_outputs,
        cuda=cuda,
    )
    pair_body, pair_inputs = _emit(
        pair,
        function_name="gfn2_aes2_pair_energy_tensor",
        outputs=("energy",),
        cuda=cuda,
    )
    pair_potential_outputs = tuple(pair_potential.program.outputs)
    pair_potential_body, pair_potential_inputs = _emit(
        pair_potential.program,
        function_name="gfn2_aes2_pair_potential_tensor",
        outputs=pair_potential_outputs,
        cuda=cuda,
    )
    pair_geometry_outputs = tuple(pair_geometry.program.outputs)
    pair_geometry_body, pair_geometry_inputs = _emit(
        pair_geometry.program,
        function_name="gfn2_aes2_pair_geometry_vjp_tensor",
        outputs=pair_geometry_outputs,
        cuda=cuda,
    )
    pair_compose_outputs = tuple(pair_compose.outputs)
    pair_compose_body, pair_compose_inputs = _emit(
        pair_compose,
        function_name="gfn2_aes2_pair_vjp_compose_tensor",
        outputs=pair_compose_outputs,
        cuda=cuda,
    )

    qualifier = "__device__ inline" if cuda else "inline"
    math = "std::"

    radius_call = _call(
        "gfn2_aes2_radius_from_fraction_tensor",
        radius_inputs,
        ("radius", "cn_derivative"),
        {"base_radius": "base_radius", "fraction": "fraction"},
        {"radius": "result.radius", "cn_derivative": "result.cn_derivative"},
    )
    kernel_call = _call(
        "gfn2_aes2_pair_kernels_tensor",
        kernel_inputs,
        ("kernel3", "kernel5"),
        {"distance": "distance", "radius": "radius"},
        {"kernel3": "result.kernel3", "kernel5": "result.kernel5"},
    )

    onsite_map = {
        "dipole_kernel": "dipole_kernel",
        "quadrupole_kernel": "quadrupole_kernel",
        **{name: f"dipole[{i}]" for i, name in enumerate(DIPOLE_NAMES)},
        **{name: f"quadrupole[{i}]" for i, name in enumerate(QUADRUPOLE_NAMES)},
        "bar_energy": "1.0",
    }
    onsite_energy_call = _call(
        "gfn2_aes2_onsite_energy_tensor",
        onsite_inputs,
        ("energy",),
        onsite_map,
        {"energy": "energy"},
    )
    onsite_outputs = {
        **{f"bar_{name}": f"result.dipole[{i}]" for i, name in enumerate(DIPOLE_NAMES)},
        **{
            f"bar_{name}": f"result.quadrupole[{i}]"
            for i, name in enumerate(QUADRUPOLE_NAMES)
        },
    }
    onsite_potential_call = _call(
        "gfn2_aes2_onsite_potential_tensor",
        onsite_vjp_inputs,
        onsite_vjp_outputs,
        onsite_map,
        onsite_outputs,
    )

    pair_map = {
        "dx": "dx",
        "dy": "dy",
        "dz": "dz",
        "kernel3": "kernel3",
        "kernel5": "kernel5",
        "first_charge": "first_charge",
        "second_charge": "second_charge",
        **{name: f"first_dipole[{i}]" for i, name in enumerate(FIRST_DIPOLE_NAMES)},
        **{name: f"second_dipole[{i}]" for i, name in enumerate(SECOND_DIPOLE_NAMES)},
        **{
            name: f"first_quadrupole[{i}]"
            for i, name in enumerate(FIRST_QUADRUPOLE_NAMES)
        },
        **{
            name: f"second_quadrupole[{i}]"
            for i, name in enumerate(SECOND_QUADRUPOLE_NAMES)
        },
        "bar_energy": "1.0",
    }
    pair_energy_call = _call(
        "gfn2_aes2_pair_energy_tensor",
        pair_inputs,
        ("energy",),
        pair_map,
        {"energy": "energy"},
    )
    pair_potential_output_map = {
        "bar_first_charge": "result.first_charge",
        "bar_second_charge": "result.second_charge",
        **{
            f"bar_{name}": f"result.first_dipole[{i}]"
            for i, name in enumerate(FIRST_DIPOLE_NAMES)
        },
        **{
            f"bar_{name}": f"result.second_dipole[{i}]"
            for i, name in enumerate(SECOND_DIPOLE_NAMES)
        },
        **{
            f"bar_{name}": f"result.first_quadrupole[{i}]"
            for i, name in enumerate(FIRST_QUADRUPOLE_NAMES)
        },
        **{
            f"bar_{name}": f"result.second_quadrupole[{i}]"
            for i, name in enumerate(SECOND_QUADRUPOLE_NAMES)
        },
    }
    pair_potential_call = _call(
        "gfn2_aes2_pair_potential_tensor",
        pair_potential_inputs,
        pair_potential_outputs,
        pair_map,
        pair_potential_output_map,
    )

    geometry_outputs = {
        "bar_dx": "bar_dx",
        "bar_dy": "bar_dy",
        "bar_dz": "bar_dz",
        "bar_kernel3": "bar_kernel3",
        "bar_kernel5": "bar_kernel5",
    }
    pair_geometry_call = _call(
        "gfn2_aes2_pair_geometry_vjp_tensor",
        pair_geometry_inputs,
        pair_geometry_outputs,
        pair_map,
        geometry_outputs,
    )
    kernel_vjp_map = {
        "distance": "distance",
        "radius": "average_radius",
        "bar_kernel3": "bar_kernel3",
        "bar_kernel5": "bar_kernel5",
    }
    kernel_vjp_output_map = {
        "bar_distance": "bar_distance",
        "bar_radius": "bar_radius",
    }
    kernel_vjp_call = _call(
        "gfn2_aes2_pair_kernel_vjp_tensor",
        kernel_vjp_inputs,
        kernel_vjp_outputs,
        kernel_vjp_map,
        kernel_vjp_output_map,
    )
    compose_map = {
        "dx": "dx",
        "dy": "dy",
        "dz": "dz",
        "distance": "distance",
        "bar_dx": "bar_dx",
        "bar_dy": "bar_dy",
        "bar_dz": "bar_dz",
        "bar_distance": "bar_distance",
        "bar_radius": "bar_radius",
        "first_radius_cn_derivative": "first_radius_cn_derivative",
        "second_radius_cn_derivative": "second_radius_cn_derivative",
    }
    compose_output_map = {
        "gradient_x": "result.gradient[0]",
        "gradient_y": "result.gradient[1]",
        "gradient_z": "result.gradient[2]",
        "first_cn_adjoint": "result.first_cn_adjoint",
        "second_cn_adjoint": "result.second_cn_adjoint",
    }
    compose_call = _call(
        "gfn2_aes2_pair_vjp_compose_tensor",
        pair_compose_inputs,
        pair_compose_outputs,
        compose_map,
        compose_output_map,
    )

    return f"""// Generated by tools/generate_gfn2_aes2_native.py from TensorIR; do not edit.
#pragma once

#include <cmath>

namespace vibeqc::xtb::generated {{

inline constexpr const char* gfn2_aes2_runtime_version = "{GFN2_AES2_RUNTIME_VERSION}";
inline constexpr double gfn2_aes2_multipole_max_radius = 5.0;
inline constexpr const char* gfn2_aes2_radius_logical_hash = "{radius.logical_hash}";
inline constexpr const char* gfn2_aes2_kernel_logical_hash = "{kernels.logical_hash}";
inline constexpr const char* gfn2_aes2_kernel_vjp_hash = "{kernel_vjp.derivative_hash}";
inline constexpr const char* gfn2_aes2_onsite_energy_logical_hash = "{onsite.logical_hash}";
inline constexpr const char* gfn2_aes2_onsite_potential_vjp_hash = "{onsite_vjp.derivative_hash}";
inline constexpr const char* gfn2_aes2_pair_energy_logical_hash = "{pair.logical_hash}";
inline constexpr const char* gfn2_aes2_pair_potential_vjp_hash = "{pair_potential.derivative_hash}";
inline constexpr const char* gfn2_aes2_pair_geometry_vjp_hash = "{pair_geometry.derivative_hash}";
inline constexpr const char* gfn2_aes2_pair_compose_logical_hash = "{pair_compose.logical_hash}";

struct Gfn2AES2RadiusResult {{
  double radius = 0.0;
  double cn_derivative = 0.0;
}};
struct Gfn2AES2KernelResult {{
  double kernel3 = 0.0;
  double kernel5 = 0.0;
}};
struct Gfn2AES2OnsitePotentialResult {{
  double dipole[3]{{}};
  double quadrupole[6]{{}};
}};
struct Gfn2AES2PairPotentialResult {{
  double first_charge = 0.0;
  double second_charge = 0.0;
  double first_dipole[3]{{}};
  double second_dipole[3]{{}};
  double first_quadrupole[6]{{}};
  double second_quadrupole[6]{{}};
}};
struct Gfn2AES2PairVjpResult {{
  double gradient[3]{{}};
  double first_cn_adjoint = 0.0;
  double second_cn_adjoint = 0.0;
}};

{radius_body}
{kernel_body}
{kernel_vjp_body}
{onsite_body}
{onsite_vjp_body}
{pair_body}
{pair_potential_body}
{pair_geometry_body}
{pair_compose_body}

{qualifier} bool evaluate_gfn2_aes2_radius(
    double coordination_number, double base_radius, double valence_cn,
    Gfn2AES2RadiusResult& result) noexcept {{
  if (!{math}isfinite(coordination_number) || !{math}isfinite(base_radius) ||
      !{math}isfinite(valence_cn)) return false;
  const double argument =
      {GFN2_AES2_MULTIPOLE_KEXP!r} *
      (coordination_number - valence_cn - {GFN2_AES2_MULTIPOLE_SHIFT!r});
  double fraction = 0.0;
  if (argument >= 0.0) {{
    const double exponential = {math}exp(-argument);
    fraction = 1.0 / (1.0 + exponential);
  }} else {{
    const double exponential = {math}exp(argument);
    fraction = exponential / (1.0 + exponential);
  }}
  return {radius_call};
}}

{qualifier} bool evaluate_gfn2_aes2_pair_kernels(
    double distance, double radius, Gfn2AES2KernelResult& result) noexcept {{
  return {kernel_call};
}}

{qualifier} bool evaluate_gfn2_aes2_onsite_energy(
    double dipole_kernel, double quadrupole_kernel, const double* dipole,
    const double* quadrupole, double& energy) noexcept {{
  return {onsite_energy_call};
}}

{qualifier} bool evaluate_gfn2_aes2_onsite_potential(
    double dipole_kernel, double quadrupole_kernel, const double* dipole,
    const double* quadrupole, Gfn2AES2OnsitePotentialResult& result) noexcept {{
  return {onsite_potential_call};
}}

{qualifier} bool evaluate_gfn2_aes2_pair_energy(
    double dx, double dy, double dz, double kernel3, double kernel5,
    double first_charge, double second_charge, const double* first_dipole,
    const double* second_dipole, const double* first_quadrupole,
    const double* second_quadrupole, double& energy) noexcept {{
  return {pair_energy_call};
}}

{qualifier} bool evaluate_gfn2_aes2_pair_potential(
    double dx, double dy, double dz, double kernel3, double kernel5,
    double first_charge, double second_charge, const double* first_dipole,
    const double* second_dipole, const double* first_quadrupole,
    const double* second_quadrupole, Gfn2AES2PairPotentialResult& result) noexcept {{
  return {pair_potential_call};
}}

{qualifier} bool evaluate_gfn2_aes2_pair_vjp(
    double dx, double dy, double dz, double kernel3, double kernel5,
    double average_radius, double first_radius_cn_derivative,
    double second_radius_cn_derivative, double first_charge, double second_charge,
    const double* first_dipole, const double* second_dipole,
    const double* first_quadrupole, const double* second_quadrupole,
    Gfn2AES2PairVjpResult& result) noexcept {{
  const double distance = {math}hypot({math}hypot(dx, dy), dz);
  if (!(distance > 0.0) || !{math}isfinite(distance) ||
      !(average_radius > 0.0) || !{math}isfinite(average_radius)) return false;
  double bar_dx = 0.0, bar_dy = 0.0, bar_dz = 0.0;
  double bar_kernel3 = 0.0, bar_kernel5 = 0.0;
  if (!{pair_geometry_call}) return false;
  double bar_distance = 0.0, bar_radius = 0.0;
  if (!{kernel_vjp_call}) return false;
  return {compose_call};
}}

}}  // namespace vibeqc::xtb::generated
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-output", type=Path)
    parser.add_argument("--cuda-output", type=Path)
    args = parser.parse_args()
    if args.cpu_output is None and args.cuda_output is None:
        parser.error("at least one output is required")
    if args.cpu_output is not None:
        args.cpu_output.parent.mkdir(parents=True, exist_ok=True)
        args.cpu_output.write_text(_header(cuda=False), encoding="utf-8")
    if args.cuda_output is not None:
        args.cuda_output.parent.mkdir(parents=True, exist_ok=True)
        args.cuda_output.write_text(_header(cuda=True), encoding="utf-8")


if __name__ == "__main__":
    main()
