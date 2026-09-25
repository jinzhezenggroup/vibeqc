"""Generate the host/device registry for qualified split global-hybrid CUDA XC."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.xc import libxc_bulk
from vibeqc_compiler.xc.libxc_maple import MapleImportError

from tools.generate_xc_split_hybrid_cuda import emit_split_hybrid_device_body
from tools.libxc_split_hybrid import build_split_global_hybrid

MANIFEST = ROOT / "manifests" / "cuda_split_hybrids.json"
GGA_CODE_BASE = 0x10000
MGGA_CODE_BASE = 0x20000


def _manifest_methods(path: Path = MANIFEST) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "vibeqc.cuda-split-global-hybrids"
        or payload.get("schema_version") != 1
    ):
        raise MapleImportError("split-hybrid CUDA manifest has unsupported schema")
    methods = payload.get("methods")
    if (
        not isinstance(methods, list)
        or not methods
        or any(not isinstance(name, str) or not name.strip() for name in methods)
        or len(set(methods)) != len(methods)
    ):
        raise MapleImportError(
            "split-hybrid CUDA manifest requires unique method names"
        )
    return tuple(methods)


def _records() -> dict[str, dict]:
    return {
        record["name"]: record for record in libxc_bulk.read_catalog()["registrations"]
    }


def _method_code(identifier: str, family: str, exchange_registration: str) -> int:
    try:
        libxc_id = _records()[exchange_registration]["id"]
    except KeyError as error:
        raise MapleImportError(
            f"split-hybrid exchange registration is absent: {exchange_registration}"
        ) from error
    if type(libxc_id) is not int or not 0 < libxc_id < 0x10000:
        raise MapleImportError("split-hybrid Libxc ID exceeds the encoded CUDA range")
    base = {"gga": GGA_CODE_BASE, "mgga": MGGA_CODE_BASE}.get(family)
    if base is None:
        raise MapleImportError(f"unsupported split-hybrid CUDA family: {family}")
    return base | libxc_id


def registry_entries(path: Path = MANIFEST) -> tuple[dict[str, object], ...]:
    entries = []
    codes = set()
    for identifier in _manifest_methods(path):
        method = build_split_global_hybrid(identifier)
        if method.exchange.family != method.correlation.family:
            raise MapleImportError(f"split-hybrid family mismatch: {identifier}")
        body, type_name, features = emit_split_hybrid_device_body(identifier)
        code = _method_code(
            identifier, method.exchange.family, method.exchange_registration
        )
        if code in codes:
            raise MapleImportError(f"duplicate split-hybrid CUDA code: {code:#x}")
        codes.add(code)
        function_stem = re.sub(r"[^a-z0-9]+", "_", identifier.lower()).strip("_")
        entries.append(
            {
                "identifier": identifier,
                "family": method.exchange.family,
                "code": code,
                "body": body,
                "type_name": type_name,
                "function_name": f"{function_stem}_device",
                "features": features,
                "exchange_registration": method.exchange_registration,
                "correlation_registration": method.correlation_registration,
                "exact_exchange_numerator": method.exact_exchange.numerator,
                "exact_exchange_denominator": method.exact_exchange.denominator,
            }
        )
    return tuple(entries)


def emit_registry(path: Path = MANIFEST) -> str:
    entries = registry_entries(path)
    code_constants = "\n".join(
        f"inline constexpr std::uint32_t k{entry['type_name'].removesuffix('DeviceValue')}FunctionalCode = "
        f"0x{entry['code']:x}U;"
        for entry in entries
    )
    registration_cases = "\n".join(
        f"    case 0x{entry['code']:x}U:" for entry in entries
    )
    mgga_cases = "\n".join(
        f"    case 0x{entry['code']:x}U:"
        for entry in entries
        if entry["family"] == "mgga"
    )
    component_cases = []
    for entry in entries:
        component_cases.extend(
            [
                "  if ((first == \""
                f"{entry['exchange_registration']}"
                "\" && second == \""
                f"{entry['correlation_registration']}"
                "\") || (first == \""
                f"{entry['correlation_registration']}"
                "\" && second == \""
                f"{entry['exchange_registration']}"
                "\"))",
                f"    return 0x{entry['code']:x}U;",
            ]
        )
    composition_cases = []
    for entry in entries:
        composition_cases.extend(
            [
                f"    case 0x{entry['code']:x}U:",
                "      return {"
                f"{entry['exact_exchange_numerator']}U, "
                f"{entry['exact_exchange_denominator']}U, true"
                "};",
            ]
        )
    device_bodies = "\n".join(str(entry["body"]).rstrip("\n") for entry in entries)
    dispatch_cases = []
    for entry in entries:
        feature_count = len(entry["features"])
        args = ["rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb"]
        if feature_count == 7:
            args.extend(("tau_a", "tau_b"))
        dispatch_cases.extend(
            [
                f"    case 0x{entry['code']:x}U: {{",
                f"      const auto raw = {entry['function_name']}({', '.join(args)});",
                "      out.energy_density = raw.energy_density;",
                f"      for (unsigned i = 0; i < {feature_count}; ++i)",
                "        out.feature_derivative[i] = raw.feature_derivative[i];",
                "      out.matched = true;",
                "      return out;",
                "    }",
            ]
        )
    return "\n".join(
        [
            "// Generated from manifests/cuda_split_hybrids.json; do not edit.",
            "#pragma once",
            "#include <cstdint>",
            "#if !defined(__CUDA_ARCH__)",
            "#include <string_view>",
            "#endif",
            "#if defined(__CUDACC__)",
            "#include <cmath>",
            "#define VIBEQC_SPLIT_HYBRID_HD __host__ __device__",
            "#else",
            "#define VIBEQC_SPLIT_HYBRID_HD",
            "#endif",
            "namespace vibeqc::dft::generated {",
            "inline constexpr std::uint32_t kSplitHybridGgaCodeBase = 0x10000U;",
            "inline constexpr std::uint32_t kSplitHybridMggaCodeBase = 0x20000U;",
            "inline constexpr std::uint32_t kSplitHybridFamilyMask = 0xf0000U;",
            code_constants,
            "#if !defined(__CUDA_ARCH__)",
            "inline constexpr std::uint32_t split_hybrid_functional_code(",
            "    std::string_view first, std::string_view second) noexcept {",
            *component_cases,
            "  return 0U;",
            "}",
            "#endif",
            "VIBEQC_SPLIT_HYBRID_HD inline constexpr bool split_hybrid_registered(",
            "    std::uint32_t functional) noexcept {",
            "  switch (functional) {",
            registration_cases,
            "      return true;",
            "    default:",
            "      return false;",
            "  }",
            "}",
            "VIBEQC_SPLIT_HYBRID_HD inline constexpr bool split_hybrid_is_mgga(",
            "    std::uint32_t functional) noexcept {",
            "  switch (functional) {",
            mgga_cases,
            "      return true;",
            "    default:",
            "      return false;",
            "  }",
            "}",
            "struct SplitHybridComposition {",
            "  std::uint32_t exact_exchange_numerator{};",
            "  std::uint32_t exact_exchange_denominator{};",
            "  bool matched{};",
            "};",
            "VIBEQC_SPLIT_HYBRID_HD inline constexpr SplitHybridComposition",
            "split_hybrid_composition(std::uint32_t functional) noexcept {",
            "  switch (functional) {",
            *composition_cases,
            "    default:",
            "      return {};",
            "  }",
            "}",
            "#if defined(__CUDACC__)",
            device_bodies,
            "struct SplitHybridRegistryValue {",
            "  double energy_density{};",
            "  double feature_derivative[7]{};",
            "  bool matched{};",
            "};",
            "__device__ inline SplitHybridRegistryValue evaluate_split_hybrid(",
            "    std::uint32_t functional, double rho_a, double rho_b,",
            "    double sigma_aa, double sigma_ab, double sigma_bb,",
            "    double tau_a, double tau_b) {",
            "  SplitHybridRegistryValue out{};",
            "  switch (functional) {",
            *dispatch_cases,
            "    default:",
            "      return out;",
            "  }",
            "}",
            "#endif",
            "}  // namespace vibeqc::dft::generated",
            "#undef VIBEQC_SPLIT_HYBRID_HD",
            "",
        ]
    )


def write_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    write_if_changed(args.output, emit_registry(args.manifest))


if __name__ == "__main__":
    main()
