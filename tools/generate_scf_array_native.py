"""Generate native CPU SCF helpers from validated SCF TensorIR equations."""

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

# Build-time code generation must work with CMake's minimal Python interpreter,
# which intentionally need not provide NumPy. Avoid tensor.__init__ (and its
# validation/interpreter imports) while loading only the build-only IR modules.
if "vibeqc_compiler.tensor" not in sys.modules:
    import vibeqc_compiler

    tensor_path = ROOT / "python" / "vibeqc_compiler" / "tensor"
    tensor_package = types.ModuleType("vibeqc_compiler.tensor")
    tensor_package.__path__ = [str(tensor_path)]
    tensor_package.__package__ = "vibeqc_compiler.tensor"
    sys.modules["vibeqc_compiler.tensor"] = tensor_package
    vibeqc_compiler.tensor = tensor_package

from vibeqc_compiler.array_api.scf import density_program, weighted_density_program
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.tensor.scf import (
    diis_extrapolation_program,
    diis_gram_program,
    hf_force_program,
)


def _kind_signature(node: typing.Any) -> tuple[str, ...]:
    return tuple(index.space.kind for index in node.spec.indices)


def template_hash(program: typing.Any) -> str:
    """Shape-independent identity for the admitted frontend/TensorIR topology."""

    def signature(node: typing.Any) -> dict[str, typing.Any]:
        attrs: dict[str, typing.Any] = {}
        for key in ("coefficient", "labels", "output"):
            if key in node.attrs:
                attrs[key] = node.attrs[key]
        if node.op == "input":
            attrs["name"] = node.attrs["name"]
        return {
            "op": node.op,
            "kinds": _kind_signature(node),
            "inputs": tuple(signature(value) for value in node.inputs),
            "attrs": attrs,
        }

    return canonical_hash(
        {
            "outputs": {
                name: signature(node) for name, node in sorted(program.outputs.items())
            }
        }
    )


def _input_name(node: typing.Any) -> str | None:
    return node.attrs.get("name") if node.op == "input" else None


def _validate_common(program: typing.Any, output_name: str) -> typing.Any:
    if program.provenance.get("construction") != "array_frontend":
        raise ValueError("SCF native generation requires Array frontend provenance")
    if tuple(program.outputs) != (output_name,):
        raise ValueError("SCF Array program must have one expected output")
    contraction = program.outputs[output_name]
    if contraction.op != "einsum" or contraction.attrs.get("coefficient") != (1, 1):
        raise ValueError("SCF Array output must be one unit-coefficient einsum")
    if _kind_signature(contraction) != ("batch", "spin", "ao", "ao"):
        raise ValueError("SCF Array output domains changed")
    labels = contraction.attrs["labels"]
    output = contraction.attrs["output"]
    # This native schedule is specialized to row-major AO/orbital storage.
    # Matching domain kinds alone cannot authorize a different operand layout.
    expected_labels = ((0, 1, 2, 3), (0, 1, 3), (0, 1, 4, 3))
    if labels != expected_labels or output != (0, 1, 2, 4):
        raise ValueError("SCF Array contraction label topology changed")
    if tuple(_kind_signature(node) for node in contraction.inputs) != (
        ("batch", "spin", "ao", "orbital"),
        ("batch", "spin", "orbital"),
        ("batch", "spin", "ao", "orbital"),
    ):
        raise ValueError("SCF Array operand layout changed")
    if contraction.inputs[0] is not contraction.inputs[2]:
        raise ValueError("SCF Array coefficients must share one input topology")
    domains: dict[int, str] = {}
    for operand, operand_labels in zip(contraction.inputs, labels, strict=True):
        if len(operand_labels) != len(operand.spec.indices):
            raise ValueError("SCF Array einsum label rank changed")
        for label, index in zip(operand_labels, operand.spec.indices, strict=True):
            kind = index.space.kind
            if domains.setdefault(label, kind) != kind:
                raise ValueError("SCF Array einsum crosses scientific index spaces")
    if tuple(domains[label] for label in output) != ("batch", "spin", "ao", "ao"):
        raise ValueError("SCF Array einsum output labels changed")
    reduced = tuple(label for label in domains if label not in output)
    if len(reduced) != 1 or domains[reduced[0]] != "orbital":
        raise ValueError("SCF Array density must reduce exactly one orbital axis")
    return contraction


def _validate_density(program: typing.Any) -> None:
    contraction = _validate_common(program, "density")
    names = tuple(_input_name(node) for node in contraction.inputs)
    if names != ("coefficients", "occupations", "coefficients"):
        raise ValueError("SCF density operand topology changed")


def _validate_weighted(program: typing.Any) -> None:
    contraction = _validate_common(program, "weighted_density")
    first, weights, third = contraction.inputs
    if (
        _input_name(first) != "coefficients"
        or first is not third
        or weights.op != "multiply"
    ):
        raise ValueError("SCF weighted-density contraction topology changed")
    names = {_input_name(node) for node in weights.inputs}
    if names != {"occupations", "orbital_energies"}:
        raise ValueError("SCF weighted-density weights topology changed")
    if _kind_signature(weights) != ("batch", "spin", "orbital"):
        raise ValueError("SCF weighted-density weight domains changed")



def _validate_hf_force(program: typing.Any, spin_count: int) -> None:
    if tuple(program.outputs) != ("forces",):
        raise ValueError("SCF HF-force program must have one expected output")
    force = program.outputs["forces"]
    if force.op != "add" or force.attrs.get("coefficients") != (
        (-1, 1),
        (-1, 1),
        (-1, 1),
        (1, 1),
    ):
        raise ValueError("SCF HF-force final sign topology changed")
    if _kind_signature(force) != ("batch", "coordinate"):
        raise ValueError("SCF HF-force output domains changed")
    if tuple(_input_name(node) for node in force.inputs[:2]) != (
        "nuclear_repulsion_derivative",
        "two_electron",
    ):
        raise ValueError("SCF HF-force scalar-source topology changed")

    one_electron, pulay = force.inputs[2:]
    for name, contraction, left_name, right_name in (
        ("one-electron", one_electron, "density", "hcore_derivative"),
        ("Pulay", pulay, "weighted_density", "overlap_derivative"),
    ):
        if (
            contraction.op != "einsum"
            or contraction.attrs.get("coefficient") != (1, 1)
            or contraction.attrs.get("labels") != ((0, 1, 2, 3), (0, 4, 2, 3))
            or contraction.attrs.get("output") != (0, 4)
        ):
            raise ValueError(f"SCF HF-force {name} contraction topology changed")
        if tuple(_input_name(node) for node in contraction.inputs) != (
            left_name,
            right_name,
        ):
            raise ValueError(f"SCF HF-force {name} operands changed")
        if tuple(_kind_signature(node) for node in contraction.inputs) != (
            ("batch", "spin", "ao", "ao"),
            ("batch", "coordinate", "ao", "ao"),
        ):
            raise ValueError(f"SCF HF-force {name} operand layout changed")
    if force.inputs[2].inputs[0].spec.indices[1].space.size != spin_count:
        raise ValueError("SCF HF-force spin layout changed")

def _validate_diis_gram(program: typing.Any) -> None:
    if tuple(program.outputs) != ("gram",):
        raise ValueError("SCF DIIS Gram program must have one expected output")
    contraction = program.outputs["gram"]
    if contraction.op != "einsum" or contraction.attrs.get("coefficient") != (1, 1):
        raise ValueError("SCF DIIS Gram must be one unit-coefficient einsum")
    if _kind_signature(contraction) != ("batch", "history", "history"):
        raise ValueError("SCF DIIS Gram output domains changed")
    if contraction.attrs.get("labels") != (
        (0, 1, 2, 3, 4),
        (0, 5, 2, 3, 4),
    ) or contraction.attrs.get("output") != (0, 1, 5):
        raise ValueError("SCF DIIS Gram contraction topology changed")
    if (
        len(contraction.inputs) != 2
        or contraction.inputs[0] is not contraction.inputs[1]
    ):
        raise ValueError(
            "SCF DIIS Gram must contract one residual-history input with itself"
        )
    residual = contraction.inputs[0]
    if _input_name(residual) != "residual_history" or _kind_signature(residual) != (
        "batch",
        "history",
        "spin",
        "ao",
        "ao",
    ):
        raise ValueError("SCF DIIS Gram residual layout changed")


def _validate_diis_extrapolation(program: typing.Any) -> None:
    if tuple(program.outputs) != ("effective_fock",):
        raise ValueError("SCF DIIS extrapolation program must have one expected output")
    contraction = program.outputs["effective_fock"]
    if contraction.op != "einsum" or contraction.attrs.get("coefficient") != (1, 1):
        raise ValueError("SCF DIIS extrapolation must be one unit-coefficient einsum")
    if _kind_signature(contraction) != ("batch", "spin", "ao", "ao"):
        raise ValueError("SCF DIIS extrapolation output domains changed")
    if contraction.attrs.get("labels") != (
        (0, 1, 2, 3, 4),
        (0, 1),
    ) or contraction.attrs.get("output") != (0, 2, 3, 4):
        raise ValueError("SCF DIIS extrapolation contraction topology changed")
    if tuple(_input_name(node) for node in contraction.inputs) != (
        "fock_history",
        "diis_coefficients",
    ):
        raise ValueError("SCF DIIS extrapolation operand topology changed")
    if tuple(_kind_signature(node) for node in contraction.inputs) != (
        ("batch", "history", "spin", "ao", "ao"),
        ("batch", "history"),
    ):
        raise ValueError("SCF DIIS extrapolation operand layout changed")


def native_header() -> str:
    density = density_program(1, 3, orbital_count=2)
    weighted = weighted_density_program(1, 3, orbital_count=2)
    restricted_force = hf_force_program(1, 2, spin_count=1, coordinate_count=3)
    unrestricted_force = hf_force_program(1, 2, spin_count=2, coordinate_count=3)
    diis_gram = diis_gram_program(1, 3, 2, spin_count=2)
    diis_extrapolation = diis_extrapolation_program(1, 3, 2, spin_count=2)
    _validate_density(density)
    _validate_weighted(weighted)
    _validate_hf_force(restricted_force, 1)
    _validate_hf_force(unrestricted_force, 2)
    _validate_diis_gram(diis_gram)
    _validate_diis_extrapolation(diis_extrapolation)
    density_hash = template_hash(density)
    weighted_hash = template_hash(weighted)
    force_hash = template_hash(restricted_force)
    if force_hash != template_hash(unrestricted_force):
        raise ValueError("SCF HF-force template must be spin-shape independent")
    diis_gram_hash = template_hash(diis_gram)
    diis_extrapolation_hash = template_hash(diis_extrapolation)
    return f"""// Generated by tools/generate_scf_array_native.py from SCF TensorIR equations.
#pragma once
#include <array>
#include <cstddef>
namespace vibeqc::scf::generated {{
inline constexpr const char* density_array_template_hash = "{density_hash}";
inline constexpr const char* weighted_density_array_template_hash = "{weighted_hash}";
inline constexpr const char* hf_force_tensor_template_hash = "{force_hash}";
inline constexpr const char* diis_gram_tensor_template_hash = "{diis_gram_hash}";
inline constexpr const char* diis_extrapolation_tensor_template_hash = "{diis_extrapolation_hash}";

inline void density_from_orbitals(double* output, const double* coefficients,
                                  std::size_t nbf, std::size_t coefficient_stride,
                                  std::size_t occupied, double occupation_weight) {{
  for (std::size_t mu = 0; mu < nbf; ++mu) {{
    for (std::size_t nu = 0; nu < nbf; ++nu) {{
      double value = 0.0;
      for (std::size_t orbital = 0; orbital < occupied; ++orbital) {{
        // Preserve the legacy FP64 product/accumulation order: the occupied
        // factor cache uses bitwise density witnesses.
        value += occupation_weight * coefficients[mu * coefficient_stride + orbital] *
                 coefficients[nu * coefficient_stride + orbital];
      }}
      output[mu * nbf + nu] = value;
    }}
  }}
}}

inline void weighted_density_from_orbitals(double* output, const double* coefficients,
                                           const double* orbital_energies, std::size_t nbf,
                                           std::size_t coefficient_stride, std::size_t occupied,
                                           double occupation_weight) {{
  for (std::size_t mu = 0; mu < nbf; ++mu) {{
    for (std::size_t nu = 0; nu < nbf; ++nu) {{
      double value = 0.0;
      for (std::size_t orbital = 0; orbital < occupied; ++orbital) {{
        value += occupation_weight * orbital_energies[orbital] *
                 coefficients[mu * coefficient_stride + orbital] *
                 coefficients[nu * coefficient_stride + orbital];
      }}
      output[mu * nbf + nu] = value;
    }}
  }}
}}


template <std::size_t SpinCount>
inline void hf_stationary_forces(
    double* output, std::size_t coordinate_count, std::size_t nbf,
    const std::array<const double*, SpinCount>& density,
    const std::array<const double*, SpinCount>& weighted_density,
    const double* hcore_derivative, const double* overlap_derivative,
    const double* two_electron, const double* nuclear_repulsion_derivative) {
  static_assert(SpinCount == 1 || SpinCount == 2);
  const std::size_t matrix = nbf * nbf;
  for (std::size_t coordinate = 0; coordinate < coordinate_count; ++coordinate) {
    const double* ds = overlap_derivative + coordinate * matrix;
    const double* dh = hcore_derivative + coordinate * matrix;
    double derivative = nuclear_repulsion_derivative[coordinate];
    for (std::size_t element = 0; element < matrix; ++element) {
      if constexpr (SpinCount == 1) {
        derivative += density[0][element] * dh[element];
        derivative -= weighted_density[0][element] * ds[element];
      } else {
        derivative += (density[0][element] + density[1][element]) * dh[element];
        derivative -=
            (weighted_density[0][element] + weighted_density[1][element]) * ds[element];
      }
    }
    derivative += two_electron[coordinate];
    output[coordinate] = -derivative;
  }
}

template <class History>
inline void diis_gram(double* output, std::size_t output_stride, const History& residual_history,
                      std::size_t history_size, std::size_t vector_size) {{
  for (std::size_t i = 0; i < history_size; ++i) {{
    for (std::size_t j = 0; j < history_size; ++j) {{
      double value = 0.0;
      for (std::size_t element = 0; element < vector_size; ++element)
        value += residual_history[i][element] * residual_history[j][element];
      output[i * output_stride + j] = value;
    }}
  }}
}}

template <class History>
inline void diis_extrapolate(double* output, const History& fock_history,
                             const double* coefficients, std::size_t history_size,
                             std::size_t vector_size) {{
  for (std::size_t element = 0; element < vector_size; ++element) output[element] = 0.0;
  // Keep the historical reduction order over the chronological history.
  for (std::size_t i = 0; i < history_size; ++i)
    for (std::size_t element = 0; element < vector_size; ++element)
      output[element] += coefficients[i] * fock_history[i][element];
}}
}}  // namespace vibeqc::scf::generated
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(native_header(), encoding="utf-8")


if __name__ == "__main__":
    main()
