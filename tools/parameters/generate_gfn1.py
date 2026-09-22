#!/usr/bin/env python3
"""Generate the canonical GFN1-xTB C++ parameter table from pinned normalized data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT / "upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3/gfn1.json"
)
DEFAULT_MANIFEST = (
    ROOT
    / "upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3/gfn1_manifest.json"
)
DEFAULT_OUTPUT = ROOT / "src/xtb/gfn2_runtime/data/parameters/gfn1.hpp"

TOP_KEYS = {
    "charge",
    "coordination_number",
    "dispersion",
    "elements",
    "halogen",
    "hamiltonian",
    "meta",
    "method",
    "repulsion",
    "schema_version",
    "thirdorder",
}
ELEMENT_KEYS = {
    "arep",
    "atomic_number",
    "atomic_radius_bohr",
    "covalent_radius_bohr",
    "dkernel",
    "en",
    "gam",
    "gam3",
    "mprad",
    "mpvcn",
    "qkernel",
    "shells",
    "symbol",
    "xbond",
    "zeff",
}
SHELL_KEYS = {
    "angular_momentum",
    "coordination_number_scale",
    "index",
    "is_valence",
    "level",
    "ngauss",
    "principal_quantum_number",
    "reference_occupation",
    "shell_hubbard_scale",
    "shell_polynomial",
    "slater",
}


class ParameterError(ValueError):
    """Pinned GFN1 data cannot be represented without changing semantics."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def expect_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParameterError(f"{label} must be an object")
    if set(value) != expected:
        raise ParameterError(
            f"{label} schema mismatch: expected {sorted(expected)}, got {sorted(value)}"
        )
    return value


def number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ParameterError(f"{label} must be finite")
    return result


def validate(parameters: dict[str, Any]) -> None:
    expect_keys(parameters, TOP_KEYS, "GFN1")
    if parameters["schema_version"] != 2 or parameters["method"] != "gfn1-xtb":
        raise ParameterError("unsupported GFN1 parameter identity")

    charge = expect_keys(parameters["charge"], {"average", "gexp"}, "charge")
    if charge["average"] != "harmonic" or number(charge["gexp"], "charge.gexp") != 2.0:
        raise ParameterError("GFN1 charge semantics changed")

    coordination = expect_keys(
        parameters["coordination_number"],
        {
            "coincident_cutoff_inclusive",
            "coincident_distance_squared_cutoff_bohr2",
            "count_expression",
            "covalent_radius_scale",
            "cutoff_bohr",
            "cutoff_inclusive",
            "derivative_expression",
            "directed_factor",
            "maximum_cn_cutoff",
            "model",
            "pair_loop",
            "steepness",
        },
        "coordination_number",
    )
    if (
        coordination["model"] != "exp"
        or coordination["maximum_cn_cutoff"] is not None
        or coordination["count_expression"]
        != "1 / (1 + exp(-k * ((r_cov_i + r_cov_j) / r - 1)))"
        or coordination["derivative_expression"]
        != "(-k * (r_cov_i + r_cov_j) * expterm) / (r^2 * (expterm + 1)^2)"
    ):
        raise ParameterError("GFN1 coordination-number semantics changed")

    dispersion = expect_keys(
        parameters["dispersion"],
        {"a1", "a2", "model", "s6", "s8", "s9", "self_consistent"},
        "dispersion",
    )
    if dispersion["model"] != "d3" or dispersion["self_consistent"] is not False:
        raise ParameterError("GFN1 dispersion semantics changed")

    expect_keys(parameters["halogen"], {"damping", "radius_scale"}, "halogen")
    expect_keys(parameters["repulsion"], {"kexp", "klight"}, "repulsion")
    thirdorder = expect_keys(
        parameters["thirdorder"], {"mode", "shell_resolved"}, "thirdorder"
    )
    if thirdorder != {"mode": "atom", "shell_resolved": False}:
        raise ParameterError("GFN1 third-order semantics changed")

    hamiltonian = expect_keys(
        parameters["hamiltonian"],
        {
            "coordination_number",
            "enscale",
            "kpol",
            "pair_scale_default",
            "pair_scale_overrides",
            "shell_pair_scale",
            "wexp",
        },
        "hamiltonian",
    )
    if hamiltonian["coordination_number"] != "exp":
        raise ParameterError("GFN1 Hamiltonian CN model changed")

    elements = parameters["elements"]
    if not isinstance(elements, list) or len(elements) != 86:
        raise ParameterError("GFN1 must contain exactly 86 elements")
    shell_index = 0
    for atomic_number, element_value in enumerate(elements, 1):
        element = expect_keys(element_value, ELEMENT_KEYS, f"element[{atomic_number}]")
        if element["atomic_number"] != atomic_number:
            raise ParameterError("GFN1 elements are not in atomic-number order")
        if not isinstance(element["symbol"], str) or not element["symbol"]:
            raise ParameterError(f"element {atomic_number} has no symbol")
        shells = element["shells"]
        if not isinstance(shells, list) or not shells:
            raise ParameterError(f"element {atomic_number} has no shells")
        for local_index, shell_value in enumerate(shells):
            shell = expect_keys(shell_value, SHELL_KEYS, f"shell[{shell_index}]")
            if type(shell["index"]) is not int or shell["index"] != local_index:
                raise ParameterError(
                    "GFN1 per-element shell indices are not contiguous"
                )
            if type(shell["angular_momentum"]) is not int or shell[
                "angular_momentum"
            ] not in (0, 1, 2):
                raise ParameterError("GFN1 contains unsupported angular momentum")
            if type(shell["is_valence"]) is not bool:
                raise ParameterError("GFN1 shell is_valence must be boolean")
            for key in ("principal_quantum_number", "ngauss"):
                if type(shell[key]) is not int or not 1 <= shell[key] <= 255:
                    raise ParameterError(
                        f"GFN1 shell {key} must be a positive uint8 integer"
                    )
            for key, value in shell.items():
                if key not in {"index", "is_valence"}:
                    number(value, f"shell[{shell_index}].{key}")
            shell_index += 1
        for key in (
            "gam",
            "gam3",
            "zeff",
            "arep",
            "xbond",
            "en",
            "dkernel",
            "qkernel",
            "mprad",
            "mpvcn",
            "atomic_radius_bohr",
            "covalent_radius_bohr",
        ):
            number(element[key], f"element[{atomic_number}].{key}")
    if shell_index != 237:
        raise ParameterError(f"GFN1 shell count changed: {shell_index}")

    shell_pairs = hamiltonian["shell_pair_scale"]
    expected_pairs = {(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)}
    actual_pairs = {tuple(entry["angular_momenta"]) for entry in shell_pairs}
    if actual_pairs != expected_pairs or len(shell_pairs) != len(expected_pairs):
        raise ParameterError("GFN1 shell-pair scale schema changed")

    previous = None
    for entry in hamiltonian["pair_scale_overrides"]:
        pair = tuple(entry["atomic_numbers"])
        if len(pair) != 2 or not (1 <= pair[0] <= pair[1] <= 86):
            raise ParameterError("invalid GFN1 pair-scale override")
        if previous is not None and pair <= previous:
            raise ParameterError("GFN1 pair-scale overrides must be strictly sorted")
        number(entry["value"], "pair_scale_override.value")
        previous = pair


def cpp_float(value: Any) -> str:
    result = format(number(value, "C++ floating value"), ".17g")
    if "." not in result and "e" not in result:
        result += ".0"
    return result


def render_header(parameters: dict[str, Any], source_revision: str) -> bytes:
    elements = parameters["elements"]
    shells = [shell for element in elements for shell in element["shells"]]
    hamiltonian = parameters["hamiltonian"]
    coordination = parameters["coordination_number"]
    entries = {
        tuple(entry["angular_momenta"]): entry["value"]
        for entry in hamiltonian["shell_pair_scale"]
    }
    shell_matrix = [
        entries[tuple(sorted((first, second)))]
        for first in range(3)
        for second in range(3)
    ]
    lines = [
        "// Generated by tools/parameters/generate_gfn1.py; do not edit.",
        "// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0",
        "// Parameter source: tblite (LGPL-3.0-or-later).",
        "// Atomic inputs: mctc-lib v0.5.2 (Apache-2.0), lengths in bohr.",
        f"// tblite revision: {source_revision}",
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <type_traits>",
        "",
        "namespace xtbloom::parameters::gfn1 {",
        "",
        "inline constexpr std::uint32_t kSchemaVersion = 2u;",
        f"inline constexpr char kSourceRevision[] = {json.dumps(source_revision)};",
        f"inline constexpr std::size_t kElementCount = {len(elements)}u;",
        f"inline constexpr std::size_t kShellCount = {len(shells)}u;",
        "",
        "struct GlobalParameters {",
        "  double hamiltonian_wexp;",
        "  double hamiltonian_kpol;",
        "  double hamiltonian_enscale;",
        "  std::array<double, 9> shell_pair_scale;",
        "  double pair_scale_default;",
        "  double dispersion_s6;",
        "  double dispersion_s8;",
        "  double dispersion_a1;",
        "  double dispersion_a2;",
        "  double dispersion_s9;",
        "  double repulsion_kexp;",
        "  double repulsion_klight;",
        "  double charge_gexp;",
        "  double halogen_damping;",
        "  double halogen_radius_scale;",
        "  double coordination_steepness;",
        "  double coordination_cutoff_bohr;",
        "  double coordination_directed_factor;",
        "  double coordination_coincident_distance_squared_cutoff_bohr2;",
        "  std::uint8_t coordination_number_model;  // 1 = exp",
        "  bool coordination_has_maximum_cn_cutoff;",
        "  bool coordination_cutoff_inclusive;",
        "  bool coordination_coincident_cutoff_inclusive;",
        "  std::uint8_t charge_average;  // 1 = harmonic",
        "  std::uint8_t dispersion_model;  // 1 = D3",
        "  bool thirdorder_shell_resolved;",
        "};",
        "",
        "struct ElementParameters {",
        "  std::uint16_t shell_offset;",
        "  std::uint8_t atomic_number;",
        "  std::uint8_t shell_count;",
        "  double gam;",
        "  double gam3;",
        "  double zeff;",
        "  double arep;",
        "  double xbond;",
        "  double electronegativity;",
        "  double dipole_kernel;",
        "  double quadrupole_kernel;",
        "  double multipole_radius;",
        "  double multipole_valence_cn;",
        "  double atomic_radius_bohr;",
        "  double covalent_radius_bohr;",
        "};",
        "",
        "struct ShellParameters {",
        "  std::uint8_t principal_quantum_number;",
        "  std::uint8_t angular_momentum;",
        "  std::uint8_t gaussian_count;",
        "  bool is_valence;",
        "  double level_electronvolt;",
        "  double slater;",
        "  double reference_occupation;",
        "  double shell_polynomial;",
        "  double coordination_number_scale_electronvolt;",
        "  double shell_hubbard_scale;",
        "};",
        "",
        "struct PairScaleOverride {",
        "  std::uint8_t first_atomic_number;",
        "  std::uint8_t second_atomic_number;",
        "  double value;",
        "};",
        "",
        "static_assert(std::is_trivially_copyable_v<GlobalParameters>);",
        "static_assert(std::is_trivially_copyable_v<ElementParameters>);",
        "static_assert(std::is_trivially_copyable_v<ShellParameters>);",
        "static_assert(std::is_trivially_copyable_v<PairScaleOverride>);",
        "",
        "inline constexpr GlobalParameters kGlobal{",
        f"  {cpp_float(hamiltonian['wexp'])},",
        f"  {cpp_float(hamiltonian['kpol'])},",
        f"  {cpp_float(hamiltonian['enscale'])},",
        "  {{" + ", ".join(cpp_float(value) for value in shell_matrix) + "}},",
        f"  {cpp_float(hamiltonian['pair_scale_default'])},",
    ]
    lines.extend(
        f"  {cpp_float(parameters['dispersion'][key])},"
        for key in ("s6", "s8", "a1", "a2", "s9")
    )
    lines.extend(
        [
            f"  {cpp_float(parameters['repulsion']['kexp'])},",
            f"  {cpp_float(parameters['repulsion']['klight'])},",
            f"  {cpp_float(parameters['charge']['gexp'])},",
            f"  {cpp_float(parameters['halogen']['damping'])},",
            f"  {cpp_float(parameters['halogen']['radius_scale'])},",
            f"  {cpp_float(coordination['steepness'])},",
            f"  {cpp_float(coordination['cutoff_bohr'])},",
            f"  {cpp_float(coordination['directed_factor'])},",
            f"  {cpp_float(coordination['coincident_distance_squared_cutoff_bohr2'])},",
            "  1u,",
            "  false,",
            f"  {str(coordination['cutoff_inclusive']).lower()},",
            f"  {str(coordination['coincident_cutoff_inclusive']).lower()},",
            "  1u,",
            "  1u,",
            "  false,",
            "};",
            "",
            "inline constexpr std::array<ElementParameters, kElementCount> kElements{{",
        ]
    )
    shell_offset = 0
    for element in elements:
        values = (
            f"{shell_offset}u",
            f"{element['atomic_number']}u",
            f"{len(element['shells'])}u",
            cpp_float(element["gam"]),
            cpp_float(element["gam3"]),
            cpp_float(element["zeff"]),
            cpp_float(element["arep"]),
            cpp_float(element["xbond"]),
            cpp_float(element["en"]),
            cpp_float(element["dkernel"]),
            cpp_float(element["qkernel"]),
            cpp_float(element["mprad"]),
            cpp_float(element["mpvcn"]),
            cpp_float(element["atomic_radius_bohr"]),
            cpp_float(element["covalent_radius_bohr"]),
        )
        lines.append("  {" + ", ".join(values) + "},")
        shell_offset += len(element["shells"])
    lines.extend(
        (
            "}};",
            "",
            "inline constexpr std::array<ShellParameters, kShellCount> kShells{{",
        )
    )
    for shell in shells:
        values = (
            f"{shell['principal_quantum_number']}u",
            f"{shell['angular_momentum']}u",
            f"{shell['ngauss']}u",
            str(shell["is_valence"]).lower(),
            cpp_float(shell["level"]),
            cpp_float(shell["slater"]),
            cpp_float(shell["reference_occupation"]),
            cpp_float(shell["shell_polynomial"]),
            cpp_float(shell["coordination_number_scale"]),
            cpp_float(shell["shell_hubbard_scale"]),
        )
        lines.append("  {" + ", ".join(values) + "},")
    lines.extend(("}};", ""))
    pair_overrides = hamiltonian["pair_scale_overrides"]
    lines.append(
        "inline constexpr std::array<PairScaleOverride, "
        f"{len(pair_overrides)}u> kPairScaleOverrides{{{{"
    )
    for override in pair_overrides:
        first, second = override["atomic_numbers"]
        lines.append(f"  {{{first}u, {second}u, {cpp_float(override['value'])}}},")
    lines.extend(
        (
            "}};",
            "",
            (
                "[[nodiscard]] constexpr const ElementParameters* find_element("
                "std::uint32_t atomic_number) noexcept {"
            ),
            (
                "  return atomic_number >= 1u && atomic_number <= kElementCount ? "
                "&kElements[atomic_number - 1u] : nullptr;"
            ),
            "}",
            "",
            (
                "[[nodiscard]] constexpr double pair_scale(std::uint32_t first, "
                "std::uint32_t second) noexcept {"
            ),
            "  if (first > second) {",
            "    const std::uint32_t swap = first;",
            "    first = second;",
            "    second = swap;",
            "  }",
            "  std::size_t begin = 0u;",
            "  std::size_t end = kPairScaleOverrides.size();",
            "  while (begin < end) {",
            "    const std::size_t middle = begin + (end - begin) / 2u;",
            "    const auto& value = kPairScaleOverrides[middle];",
            "    if (value.first_atomic_number < first ||",
            (
                "        (value.first_atomic_number == first && "
                "value.second_atomic_number < second)) {"
            ),
            "      begin = middle + 1u;",
            "    } else {",
            "      end = middle;",
            "    }",
            "  }",
            "  if (begin < kPairScaleOverrides.size()) {",
            "    const auto& value = kPairScaleOverrides[begin];",
            "    if (value.first_atomic_number == first &&",
            "        value.second_atomic_number == second) return value.value;",
            "  }",
            "  return kGlobal.pair_scale_default;",
            "}",
            "",
            "}  // namespace xtbloom::parameters::gfn1",
            "",
        )
    )
    return "\n".join(lines).encode("utf-8")


def load_and_render(source: Path, manifest_path: Path) -> bytes:
    source_bytes = source.read_bytes()
    parameters = json.loads(source_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        expected_json = manifest["outputs"]["gfn1.json"]["sha256"]
        expected_header = manifest["outputs"]["gfn1.hpp"]["sha256"]
        source_revision = manifest["source"]["revision"]
    except (KeyError, TypeError) as error:
        raise ParameterError("GFN1 source manifest is incomplete") from error
    if sha256(source_bytes) != expected_json:
        raise ParameterError("GFN1 normalized JSON does not match its source manifest")
    validate(parameters)
    rendered = render_header(parameters, source_revision)
    if sha256(rendered) != expected_header:
        raise ParameterError(
            "generated GFN1 header differs from the audited upstream product; "
            "review the schema/renderer before accepting new bytes"
        )
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rendered = load_and_render(args.source, args.manifest)
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != rendered:
            raise ParameterError(
                f"generated GFN1 parameter header is stale: {args.output}"
            )
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
