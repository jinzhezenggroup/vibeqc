"""Generate CUDA device helpers for fixed-state GFN2 electronic pair science."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

# Keep build-time codegen independent of NumPy-backed interpreter packages.
import vibeqc_compiler

for package_name in ("tensor", "method"):
    qualified = f"vibeqc_compiler.{package_name}"
    if qualified not in sys.modules:
        package = types.ModuleType(qualified)
        package.__path__ = [str(ROOT / "python" / "vibeqc_compiler" / package_name)]
        package.__package__ = qualified
        sys.modules[qualified] = package
        setattr(vibeqc_compiler, package_name, package)

from vibeqc_compiler.method.gfn2_electronic_contract import (
    GFN2_DIPOLE_COMPONENTS,
    GFN2_QUADRUPOLE_COMPONENTS,
)
from vibeqc_compiler.method.gfn2_electronic_runtime import (
    GFN2_ELECTRONIC_PAIR_VERSION,
    build_gfn2_runtime_electronic_pair_primal,
    build_gfn2_runtime_electronic_pair_vjp,
    build_gfn2_runtime_overlap_vjp,
)
from vibeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp


def _device(source: str, function_name: str) -> str:
    needle = f"inline bool {function_name}("
    replacement = f"__device__ inline bool {function_name}("
    if source.count(needle) != 1:
        raise ValueError(f"expected one generated declaration for {function_name}")
    return source.replace(needle, replacement, 1)


def _primal_input_order() -> tuple[str, ...]:
    names = ["overlap", "row_scalar_potential", "column_scalar_potential"]
    for prefix, components in (
        ("dipole", GFN2_DIPOLE_COMPONENTS),
        ("quadrupole", GFN2_QUADRUPOLE_COMPONENTS),
    ):
        for component in components:
            names.extend(
                (
                    f"{prefix}_forward_{component}",
                    f"{prefix}_reverse_{component}",
                    f"{prefix}_row_potential_{component}",
                    f"{prefix}_column_potential_{component}",
                )
            )
    return tuple(names)


def _vjp_input_order() -> tuple[str, ...]:
    names = ["bar_shift", "row_scalar_potential", "column_scalar_potential"]
    for prefix, components in (
        ("dipole", GFN2_DIPOLE_COMPONENTS),
        ("quadrupole", GFN2_QUADRUPOLE_COMPONENTS),
    ):
        for component in components:
            names.extend(
                (
                    f"{prefix}_row_potential_{component}",
                    f"{prefix}_column_potential_{component}",
                )
            )
    return tuple(names)


def _vjp_output_order() -> tuple[str, ...]:
    names = ["bar_overlap"]
    for prefix, components in (
        ("dipole", GFN2_DIPOLE_COMPONENTS),
        ("quadrupole", GFN2_QUADRUPOLE_COMPONENTS),
    ):
        for component in components:
            names.extend(
                (
                    f"bar_{prefix}_forward_{component}",
                    f"bar_{prefix}_reverse_{component}",
                )
            )
    return tuple(names)


def _cpp_input(name: str) -> str:
    if name == "overlap":
        return "integrals.overlap"
    if name == "bar_shift":
        return "bar_shift"
    if name == "row_scalar_potential":
        return "potentials.row_scalar"
    if name == "column_scalar_potential":
        return "potentials.column_scalar"
    for prefix, components in (
        ("dipole", GFN2_DIPOLE_COMPONENTS),
        ("quadrupole", GFN2_QUADRUPOLE_COMPONENTS),
    ):
        for index, component in enumerate(components):
            table = {
                f"{prefix}_forward_{component}": f"integrals.{prefix}_forward[{index}]",
                f"{prefix}_reverse_{component}": f"integrals.{prefix}_reverse[{index}]",
                f"{prefix}_row_potential_{component}": f"potentials.{prefix}_row[{index}]",
                f"{prefix}_column_potential_{component}": f"potentials.{prefix}_column[{index}]",
            }
            if name in table:
                return table[name]
    raise KeyError(name)


def _cpp_output(name: str) -> str:
    if name == "bar_overlap":
        return "adjoint.overlap"
    for prefix, components in (
        ("dipole", GFN2_DIPOLE_COMPONENTS),
        ("quadrupole", GFN2_QUADRUPOLE_COMPONENTS),
    ):
        for index, component in enumerate(components):
            if name == f"bar_{prefix}_forward_{component}":
                return f"adjoint.{prefix}_forward[{index}]"
            if name == f"bar_{prefix}_reverse_{component}":
                return f"adjoint.{prefix}_reverse[{index}]"
    raise KeyError(name)


def cuda_header() -> str:
    primal = build_gfn2_runtime_electronic_pair_primal()
    vjp = build_gfn2_runtime_electronic_pair_vjp()
    overlap_vjp = build_gfn2_runtime_overlap_vjp()
    primal_inputs = _primal_input_order()
    vjp_inputs = _vjp_input_order()
    vjp_outputs = _vjp_output_order()
    primal_source = _device(
        emit_scalar_cpp(
            primal,
            function_name="gfn2_electronic_pair_tensor",
            input_order=primal_inputs,
            output_order=("shift",),
            check_intermediates=False,
        ),
        "gfn2_electronic_pair_tensor",
    )
    vjp_source = _device(
        emit_scalar_cpp(
            vjp,
            function_name="gfn2_electronic_pair_vjp_tensor",
            input_order=vjp_inputs,
            output_order=vjp_outputs,
            check_intermediates=False,
        ),
        "gfn2_electronic_pair_vjp_tensor",
    )
    overlap_source = _device(
        emit_scalar_cpp(
            overlap_vjp,
            function_name="gfn2_electronic_overlap_vjp_tensor",
            input_order=(
                "bar_shift",
                "row_scalar_potential",
                "column_scalar_potential",
            ),
            output_order=("bar_overlap",),
            check_intermediates=False,
        ),
        "gfn2_electronic_overlap_vjp_tensor",
    )
    primal_call = ", ".join([*map(_cpp_input, primal_inputs), "shift"])
    vjp_call = ", ".join([*map(_cpp_input, vjp_inputs), *map(_cpp_output, vjp_outputs)])
    return f"""// Generated by tools/generate_gfn2_electronic_cuda.py from TensorIR; do not edit.
#pragma once

#include <cmath>

namespace vibeqc::xtb::generated {{

inline constexpr const char* gfn2_electronic_pair_version = "{GFN2_ELECTRONIC_PAIR_VERSION}";
inline constexpr const char* gfn2_electronic_pair_primal_hash = "{primal.logical_hash}";
inline constexpr const char* gfn2_electronic_pair_vjp_hash = "{vjp.logical_hash}";

struct Gfn2ElectronicPairIntegrals {{
  double overlap = 0.0;
  double dipole_forward[3]{{}};
  double dipole_reverse[3]{{}};
  double quadrupole_forward[6]{{}};
  double quadrupole_reverse[6]{{}};
}};

struct Gfn2ElectronicPairPotentials {{
  double row_scalar = 0.0;
  double column_scalar = 0.0;
  double dipole_row[3]{{}};
  double dipole_column[3]{{}};
  double quadrupole_row[6]{{}};
  double quadrupole_column[6]{{}};
}};

struct Gfn2ElectronicPairAdjoint {{
  double overlap = 0.0;
  double dipole_forward[3]{{}};
  double dipole_reverse[3]{{}};
  double quadrupole_forward[6]{{}};
  double quadrupole_reverse[6]{{}};
}};

{primal_source}
{vjp_source}
{overlap_source}
__device__ inline bool evaluate_gfn2_electronic_pair(
    const Gfn2ElectronicPairIntegrals& integrals,
    const Gfn2ElectronicPairPotentials& potentials,
    double& shift) noexcept {{
  return gfn2_electronic_pair_tensor({primal_call});
}}

__device__ inline bool evaluate_gfn2_electronic_pair_vjp(
    double bar_shift, const Gfn2ElectronicPairPotentials& potentials,
    Gfn2ElectronicPairAdjoint& adjoint) noexcept {{
  return gfn2_electronic_pair_vjp_tensor({vjp_call});
}}
__device__ inline bool evaluate_gfn2_electronic_overlap_vjp(
    double bar_shift, double row_scalar, double column_scalar,
    double& bar_overlap) noexcept {{
  return gfn2_electronic_overlap_vjp_tensor(
      bar_shift, row_scalar, column_scalar, bar_overlap);
}}

}}  // namespace vibeqc::xtb::generated
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(cuda_header(), encoding="utf-8")


if __name__ == "__main__":
    main()
