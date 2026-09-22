"""Generate shared CPU/CUDA helpers for compiler-owned GFN2 H0-force pair science."""

from __future__ import annotations

import argparse
import sys
import types
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

# Build-time generation must remain independent of NumPy-backed interpreter imports.
import vibeqc_compiler

for package_name in ("tensor", "method"):
    qualified = f"vibeqc_compiler.{package_name}"
    if qualified not in sys.modules:
        package = types.ModuleType(qualified)
        package.__path__ = [str(ROOT / "python" / "vibeqc_compiler" / package_name)]
        package.__package__ = qualified
        sys.modules[qualified] = package
        setattr(vibeqc_compiler, package_name, package)

from vibeqc_compiler.method.gfn2_h0_force_runtime import (
    GFN2_H0_FORCE_RUNTIME_VERSION,
    build_gfn2_h0_ao_update_program,
    build_gfn2_h0_distance_program,
    build_gfn2_h0_distance_vjp_program,
    build_gfn2_h0_offsite_factor_program,
    build_gfn2_h0_offsite_vjp_program,
    build_gfn2_h0_onsite_factor_program,
    build_gfn2_h0_onsite_vjp_program,
    build_gfn2_h0_pulay_seed_program,
)
from vibeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp

if typing.TYPE_CHECKING:
    from vibeqc_compiler.tensor.program import Program

PAIR_INPUT_ORDER = (
    "first_shell_level",
    "second_shell_level",
    "first_cn_scale",
    "second_cn_scale",
    "first_cn",
    "second_cn",
    "first_radius",
    "second_radius",
    "first_polynomial",
    "second_polynomial",
    "pair_scale",
    "distance",
    "bar_factor",
)
AO_INPUT_ORDER = ("density", "overlap", "factor", "overlap_adjoint", "block_weight")
DISTANCE_INPUT_ORDER = ("dx", "dy", "dz", "bar_distance")
PULAY_INPUT_ORDER = ("seed", "weighted")


def _device(source: str, function_name: str) -> str:
    needle = f"inline bool {function_name}("
    replacement = f"VIBEQC_GFN2_H0_HD inline bool {function_name}("
    if source.count(needle) != 1:
        raise ValueError(f"expected one generated declaration for {function_name}")
    return source.replace(needle, replacement, 1)


def _inputs(program: Program, canonical: tuple[str, ...]) -> tuple[str, ...]:
    live = {node.attrs["name"] for node in program.live_nodes if node.op == "input"}
    ordered = tuple(name for name in canonical if name in live)
    if set(ordered) != live:
        missing = sorted(live - set(ordered))
        raise ValueError(f"unmapped generated inputs: {missing}")
    return ordered


def _pair_cpp(name: str) -> str:
    if name == "bar_factor":
        return "bar_factor"
    return f"input.{name}"


def native_header() -> str:
    onsite = build_gfn2_h0_onsite_factor_program()
    offsite = build_gfn2_h0_offsite_factor_program()
    onsite_vjp = build_gfn2_h0_onsite_vjp_program()
    offsite_vjp = build_gfn2_h0_offsite_vjp_program()
    ao_update = build_gfn2_h0_ao_update_program()
    distance = build_gfn2_h0_distance_program()
    distance_vjp = build_gfn2_h0_distance_vjp_program()
    pulay_seed = build_gfn2_h0_pulay_seed_program()

    onsite_inputs = _inputs(onsite, PAIR_INPUT_ORDER)
    offsite_inputs = _inputs(offsite, PAIR_INPUT_ORDER)
    onsite_vjp_inputs = _inputs(onsite_vjp, PAIR_INPUT_ORDER)
    offsite_vjp_inputs = _inputs(offsite_vjp, PAIR_INPUT_ORDER)

    onsite_source = _device(
        emit_scalar_cpp(
            onsite,
            function_name="gfn2_h0_onsite_factor_tensor",
            input_order=onsite_inputs,
            output_order=("factor",),
        ),
        "gfn2_h0_onsite_factor_tensor",
    )
    offsite_source = _device(
        emit_scalar_cpp(
            offsite,
            function_name="gfn2_h0_offsite_factor_tensor",
            input_order=offsite_inputs,
            output_order=("factor",),
        ),
        "gfn2_h0_offsite_factor_tensor",
    )
    onsite_vjp_source = _device(
        emit_scalar_cpp(
            onsite_vjp,
            function_name="gfn2_h0_onsite_vjp_tensor",
            input_order=onsite_vjp_inputs,
            output_order=("bar_first_cn", "bar_second_cn"),
        ),
        "gfn2_h0_onsite_vjp_tensor",
    )
    offsite_vjp_source = _device(
        emit_scalar_cpp(
            offsite_vjp,
            function_name="gfn2_h0_offsite_vjp_tensor",
            input_order=offsite_vjp_inputs,
            output_order=("bar_first_cn", "bar_second_cn", "bar_distance"),
        ),
        "gfn2_h0_offsite_vjp_tensor",
    )
    ao_source = _device(
        emit_scalar_cpp(
            ao_update,
            function_name="gfn2_h0_ao_update_tensor",
            input_order=AO_INPUT_ORDER,
            output_order=("overlap_adjoint_updated", "block_weight_updated"),
        ),
        "gfn2_h0_ao_update_tensor",
    )

    distance_inputs = _inputs(distance, DISTANCE_INPUT_ORDER)
    distance_vjp_inputs = _inputs(distance_vjp, DISTANCE_INPUT_ORDER)
    distance_source = _device(
        emit_scalar_cpp(
            distance,
            function_name="gfn2_h0_distance_tensor",
            input_order=distance_inputs,
            output_order=("distance_squared", "distance"),
        ),
        "gfn2_h0_distance_tensor",
    )
    distance_vjp_source = _device(
        emit_scalar_cpp(
            distance_vjp,
            function_name="gfn2_h0_distance_vjp_tensor",
            input_order=distance_vjp_inputs,
            output_order=("bar_dx", "bar_dy", "bar_dz"),
        ),
        "gfn2_h0_distance_vjp_tensor",
    )
    pulay_source = _device(
        emit_scalar_cpp(
            pulay_seed,
            function_name="gfn2_h0_pulay_seed_tensor",
            input_order=PULAY_INPUT_ORDER,
            output_order=("pulay_seed",),
        ),
        "gfn2_h0_pulay_seed_tensor",
    )

    onsite_call = ", ".join([*map(_pair_cpp, onsite_inputs), "factor"])
    offsite_call = ", ".join([*map(_pair_cpp, offsite_inputs), "factor"])
    onsite_vjp_call = ", ".join(
        [
            *map(_pair_cpp, onsite_vjp_inputs),
            "adjoint.first_cn",
            "adjoint.second_cn",
        ]
    )
    offsite_vjp_call = ", ".join(
        [
            *map(_pair_cpp, offsite_vjp_inputs),
            "adjoint.first_cn",
            "adjoint.second_cn",
            "adjoint.distance",
        ]
    )
    distance_call = ", ".join(
        [
            *(name for name in distance_inputs),
            "distance_squared",
            "distance",
        ]
    )
    distance_vjp_call = ", ".join(
        [
            *(
                "bar_distance" if name == "bar_distance" else name
                for name in distance_vjp_inputs
            ),
            "adjoint.dx",
            "adjoint.dy",
            "adjoint.dz",
        ]
    )

    return f"""// Generated by tools/generate_gfn2_h0_native.py from TensorIR; do not edit.
#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define VIBEQC_GFN2_H0_HD __host__ __device__
#else
#define VIBEQC_GFN2_H0_HD
#endif

namespace vibeqc::xtb::generated {{

inline constexpr const char* gfn2_h0_force_runtime_version =
    "{GFN2_H0_FORCE_RUNTIME_VERSION}";
inline constexpr const char* gfn2_h0_onsite_factor_hash = "{onsite.logical_hash}";
inline constexpr const char* gfn2_h0_offsite_factor_hash = "{offsite.logical_hash}";
inline constexpr const char* gfn2_h0_onsite_vjp_hash = "{onsite_vjp.logical_hash}";
inline constexpr const char* gfn2_h0_offsite_vjp_hash = "{offsite_vjp.logical_hash}";
inline constexpr const char* gfn2_h0_ao_update_hash = "{ao_update.logical_hash}";
inline constexpr const char* gfn2_h0_distance_hash = "{distance.logical_hash}";
inline constexpr const char* gfn2_h0_distance_vjp_hash = "{distance_vjp.logical_hash}";
inline constexpr const char* gfn2_h0_pulay_seed_hash = "{pulay_seed.logical_hash}";

struct Gfn2H0PairInput {{
  double first_shell_level = 0.0;
  double second_shell_level = 0.0;
  double first_cn_scale = 0.0;
  double second_cn_scale = 0.0;
  double first_cn = 0.0;
  double second_cn = 0.0;
  double first_radius = 0.0;
  double second_radius = 0.0;
  double first_polynomial = 0.0;
  double second_polynomial = 0.0;
  double pair_scale = 0.0;
  double distance = 0.0;
}};

struct Gfn2H0PairAdjoint {{
  double first_cn = 0.0;
  double second_cn = 0.0;
  double distance = 0.0;
}};

struct Gfn2H0CartesianAdjoint {{
  double dx = 0.0;
  double dy = 0.0;
  double dz = 0.0;
}};

{onsite_source}
{offsite_source}
{onsite_vjp_source}
{offsite_vjp_source}
{ao_source}
{distance_source}
{distance_vjp_source}
{pulay_source}
VIBEQC_GFN2_H0_HD inline bool evaluate_gfn2_h0_onsite_factor(
    const Gfn2H0PairInput& input, double& factor) noexcept {{
  return gfn2_h0_onsite_factor_tensor({onsite_call});
}}

VIBEQC_GFN2_H0_HD inline bool evaluate_gfn2_h0_offsite_factor(
    const Gfn2H0PairInput& input, double& factor) noexcept {{
  return gfn2_h0_offsite_factor_tensor({offsite_call});
}}

VIBEQC_GFN2_H0_HD inline bool evaluate_gfn2_h0_onsite_vjp(
    double bar_factor, const Gfn2H0PairInput& input,
    Gfn2H0PairAdjoint& adjoint) noexcept {{
  return gfn2_h0_onsite_vjp_tensor({onsite_vjp_call});
}}

VIBEQC_GFN2_H0_HD inline bool evaluate_gfn2_h0_offsite_vjp(
    double bar_factor, const Gfn2H0PairInput& input,
    Gfn2H0PairAdjoint& adjoint) noexcept {{
  return gfn2_h0_offsite_vjp_tensor({offsite_vjp_call});
}}

VIBEQC_GFN2_H0_HD inline bool accumulate_gfn2_h0_ao(
    double density, double overlap, double factor,
    double& overlap_adjoint, double& block_weight) noexcept {{
  double overlap_adjoint_updated = 0.0;
  double block_weight_updated = 0.0;
  if (!gfn2_h0_ao_update_tensor(
          density, overlap, factor, overlap_adjoint, block_weight,
          overlap_adjoint_updated, block_weight_updated)) {{
    return false;
  }}
  overlap_adjoint = overlap_adjoint_updated;
  block_weight = block_weight_updated;
  return true;
}}

VIBEQC_GFN2_H0_HD inline bool evaluate_gfn2_h0_distance(
    double dx, double dy, double dz,
    double& distance_squared, double& distance) noexcept {{
  return gfn2_h0_distance_tensor({distance_call});
}}

VIBEQC_GFN2_H0_HD inline bool evaluate_gfn2_h0_distance_vjp(
    double bar_distance, double dx, double dy, double dz,
    Gfn2H0CartesianAdjoint& adjoint) noexcept {{
  return gfn2_h0_distance_vjp_tensor({distance_vjp_call});
}}

VIBEQC_GFN2_H0_HD inline bool evaluate_gfn2_h0_pulay_seed(
    double seed, double weighted, double& pulay_seed) noexcept {{
  return gfn2_h0_pulay_seed_tensor(seed, weighted, pulay_seed);
}}

}}  // namespace vibeqc::xtb::generated

#undef VIBEQC_GFN2_H0_HD
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(native_header(), encoding="utf-8")


if __name__ == "__main__":
    main()
