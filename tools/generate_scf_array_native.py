"""Generate native CPU SCF tensor helpers from the bounded Array frontend."""

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


def native_header() -> str:
    density = density_program(1, 3, orbital_count=2)
    weighted = weighted_density_program(1, 3, orbital_count=2)
    _validate_density(density)
    _validate_weighted(weighted)
    density_hash = template_hash(density)
    weighted_hash = template_hash(weighted)
    return f"""// Generated by tools/generate_scf_array_native.py from array_api.scf -> TensorIR.
#pragma once
#include <cstddef>
namespace vibeqc::scf::generated {{
inline constexpr const char* density_array_template_hash = "{density_hash}";
inline constexpr const char* weighted_density_array_template_hash = "{weighted_hash}";

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
