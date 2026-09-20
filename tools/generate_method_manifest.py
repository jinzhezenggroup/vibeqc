"""Generate public method ABI/runtime metadata from one audited manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "methods/public_methods.json"
C_IDS = ROOT / "include/vibeqc/generated_method_ids.h"
PYTHON = ROOT / "python/vibeqc/_generated_methods.py"
CPP = ROOT / "src/methods/generated_method_manifest.hpp"

FAMILIES = {
    "hartree_fock": "VIBEQC_METHOD_FAMILY_HARTREE_FOCK",
    "density_functional": "VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL",
    "coupled_cluster": "VIBEQC_METHOD_FAMILY_COUPLED_CLUSTER",
    "perturbation": "VIBEQC_METHOD_FAMILY_PERTURBATION",
    "semiempirical": "VIBEQC_METHOD_FAMILY_SEMIEMPIRICAL",
}
PROVIDERS = {
    "reserved": ("Reserved", False, False),
    "hf": ("Hf", True, True),
    "mp2": ("Mp2", True, True),
    "rccsd": ("Rccsd", True, True),
    "dft": ("Dft", True, True),
    "xtb": ("Xtb", True, False),
}
PROPERTIES = {
    "energy": "VIBEQC_PROPERTY_ENERGY",
    "forces": "VIBEQC_PROPERTY_FORCES",
}


def load_manifest() -> list[dict]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported public method manifest schema")
    methods = payload.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("public method manifest requires a non-empty methods list")

    names: set[str] = set()
    symbols: set[str] = set()
    ids: set[int] = set()
    aliases: set[str] = set()
    for method in methods:
        required = {
            "name",
            "symbol",
            "abi_id",
            "family",
            "provider",
            "properties",
            "supports_batch",
        }
        optional = {"aliases", "unavailable_reason"}
        if set(method) - (required | optional):
            raise ValueError(
                f"unknown method manifest fields for {method.get('name')!r}"
            )
        if not required <= set(method):
            raise ValueError(f"incomplete method manifest entry {method!r}")

        name = method["name"]
        symbol = method["symbol"]
        abi_id = method["abi_id"]
        if not isinstance(name, str) or not name or name != name.lower():
            raise ValueError(
                "canonical public method names must be non-empty lowercase strings"
            )
        if (
            not isinstance(symbol, str)
            or not symbol
            or not symbol.replace("_", "").isalnum()
        ):
            raise ValueError(f"invalid C/Python method symbol {symbol!r}")
        if (
            not isinstance(abi_id, int)
            or isinstance(abi_id, bool)
            or not 0 < abi_id < 2**31
        ):
            raise ValueError(
                f"method {name} requires one explicit positive int32 ABI id"
            )
        if name in names or symbol in symbols or abi_id in ids:
            raise ValueError("public method names, symbols and ABI ids must be unique")
        names.add(name)
        symbols.add(symbol)
        ids.add(abi_id)

        if method["family"] not in FAMILIES:
            raise ValueError(f"unknown method family {method['family']!r}")
        if method["provider"] not in PROVIDERS:
            raise ValueError(f"unknown native provider {method['provider']!r}")
        properties = method["properties"]
        if not isinstance(properties, list) or any(
            not isinstance(prop, str) for prop in properties
        ):
            raise ValueError(f"{name}: properties must be a list of names")
        if len(set(properties)) != len(properties):
            raise ValueError(f"{name}: public properties must be unique")
        if type(method["supports_batch"]) is not bool:
            raise ValueError(f"{name}: supports_batch must be a boolean")
        if any(prop not in PROPERTIES for prop in properties):
            raise ValueError(f"unknown public property in {name}")

        provider_executable = PROVIDERS[method["provider"]][1]
        provider_batch = PROVIDERS[method["provider"]][2]
        if provider_executable != bool(method["properties"]):
            raise ValueError(
                f"{name}: executable providers require public properties and "
                "reserved providers forbid them"
            )
        if provider_batch != bool(method["supports_batch"]):
            raise ValueError(
                f"{name}: batch capability disagrees with its native provider"
            )
        if method["provider"] == "hf" and method["family"] != "hartree_fock":
            raise ValueError(f"{name}: HF provider requires Hartree-Fock family")
        if method["provider"] == "mp2" and method["family"] != "perturbation":
            raise ValueError(f"{name}: MP2 provider requires perturbation family")
        if method["provider"] == "rccsd" and method["family"] != "coupled_cluster":
            raise ValueError(f"{name}: RCCSD provider requires coupled-cluster family")
        if method["provider"] == "dft" and method["family"] != "density_functional":
            raise ValueError(f"{name}: DFT provider requires density-functional family")
        if method["provider"] == "xtb" and method["family"] != "semiempirical":
            raise ValueError(f"{name}: xTB provider requires semiempirical family")

        method_aliases = method.get("aliases", [])
        if not isinstance(method_aliases, list) or any(
            not isinstance(alias, str) or not alias or alias != alias.lower()
            for alias in method_aliases
        ):
            raise ValueError(f"{name}: aliases must be non-empty lowercase strings")
        for alias in method_aliases:
            if alias in names or alias in aliases:
                raise ValueError(f"duplicate public method alias {alias!r}")
            aliases.add(alias)

    if aliases & names:
        raise ValueError("public method aliases cannot shadow canonical names")
    return sorted(methods, key=lambda method: method["abi_id"])


def emit_c_ids(methods: list[dict]) -> str:
    lines = [
        "// Generated by tools/generate_method_manifest.py from methods/public_methods.json.",
        "// Do not edit by hand; ABI ids are explicit manifest data and must never be renumbered implicitly.",
        "#ifndef VIBEQC_GENERATED_METHOD_IDS_H",
        "#define VIBEQC_GENERATED_METHOD_IDS_H",
        "",
        "#include <stdint.h>",
        "",
        "// clang-format off",
        "typedef int32_t vibeqc_method;",
        "enum {",
    ]
    for index, method in enumerate(methods):
        comma = "," if index + 1 < len(methods) else ""
        lines.append(f"  VIBEQC_METHOD_{method['symbol']} = {method['abi_id']}{comma}")
    lines.extend(["};", "// clang-format on", "", "#endif", ""])
    return "\n".join(lines)


def emit_python(methods: list[dict]) -> str:
    lines = [
        '"""Generated public method identity metadata; do not edit by hand."""',
        "",
        "# fmt: off",
        "from types import MappingProxyType",
        "",
    ]
    for method in methods:
        lines.append(f"METHOD_{method['symbol']} = {method['abi_id']}")

    lines.extend(["", "METHOD_CONSTANTS = MappingProxyType({"])
    for method in methods:
        lines.append(f'    "METHOD_{method["symbol"]}": METHOD_{method["symbol"]},')
    lines.extend(["})", "", "METHOD_METADATA = MappingProxyType({"])
    for method in methods:
        method_aliases = tuple(method.get("aliases", []))
        lines.append(
            f'    {method["name"]!r}: MappingProxyType({{"abi_id": '
            f'{method["abi_id"]}, "family": {method["family"]!r}, '
            f'"provider": {method["provider"]!r}, '
            f'"properties": {tuple(method["properties"])!r}, '
            f'"supports_batch": {bool(method["supports_batch"])!r}, '
            f'"aliases": {method_aliases!r}}}),'
        )
    lines.extend(["})", "", "METHOD_NAME_TO_ID = MappingProxyType({"])
    for method in methods:
        lines.append(f"    {method['name']!r}: METHOD_{method['symbol']},")
        for alias in method.get("aliases", []):
            lines.append(f"    {alias!r}: METHOD_{method['symbol']},")
    lines.extend(["})", "METHOD_ID_TO_NAME = MappingProxyType({"])
    for method in methods:
        lines.append(f"    METHOD_{method['symbol']}: {method['name']!r},")
    lines.extend(["})", ""])

    hf = [f"METHOD_{m['symbol']}" for m in methods if m["provider"] == "hf"]
    dft = [f"METHOD_{m['symbol']}" for m in methods if m["provider"] == "dft"]
    lines.append(f"HF_METHOD_IDS = frozenset(({', '.join(hf)}{',' if hf else ''}))")
    lines.append(
        f"NATIVE_DFT_METHOD_IDS = frozenset(({', '.join(dft)}{',' if dft else ''}))"
    )
    lines.extend(["# fmt: on", ""])
    return "\n".join(lines)


def cpp_properties(method: dict) -> str:
    values = [PROPERTIES[prop] for prop in method["properties"]]
    return " | ".join(values) if values else "0"


def emit_cpp(methods: list[dict]) -> str:
    lines = [
        "// Generated by tools/generate_method_manifest.py from methods/public_methods.json.",
        "// Native function pointers remain registry-owned; this file owns public identity/capability facts.",
        "#ifndef VIBEQC_METHODS_GENERATED_METHOD_MANIFEST_HPP",
        "#define VIBEQC_METHODS_GENERATED_METHOD_MANIFEST_HPP",
        "",
        "#include <array>",
        "#include <cstdint>",
        "#include <string_view>",
        "",
        '#include "vibeqc/vibeqc.h"',
        "",
        "// clang-format off",
        "namespace vibeqc::methods::generated {",
        "",
        "enum class PublicProvider : std::uint8_t { Reserved, Hf, Mp2, Rccsd, Dft, Xtb };",
        "",
        "struct MethodManifestEntry {",
        "  std::string_view name;",
        "  vibeqc_method method;",
        "  vibeqc_method_family family;",
        "  vibeqc_property_flags properties;",
        "  bool supports_batch;",
        "  PublicProvider provider;",
        "  std::string_view unavailable_reason;",
        "};",
        "",
        f"inline constexpr std::array<MethodManifestEntry, {len(methods)}> kMethodManifest{{{{",
    ]
    for method in methods:
        reason = method.get("unavailable_reason", "")
        lines.append(
            f'    {{"{method["name"]}", VIBEQC_METHOD_{method["symbol"]}, '
            f"{FAMILIES[method['family']]}, {cpp_properties(method)}, "
            f"{'true' if method['supports_batch'] else 'false'}, "
            f'PublicProvider::{PROVIDERS[method["provider"]][0]}, "{reason}"}},'
        )
    lines.extend(
        [
            "}};",
            "",
            "inline constexpr const MethodManifestEntry* find_method(",
            "    std::string_view name) noexcept {",
            "  for (const auto& method : kMethodManifest)",
            "    if (method.name == name) return &method;",
            "  return nullptr;",
            "}",
            "",
            "inline constexpr const MethodManifestEntry* find_method(",
            "    vibeqc_method value) noexcept {",
            "  for (const auto& method : kMethodManifest)",
            "    if (method.method == value) return &method;",
            "  return nullptr;",
            "}",
            "",
            "}  // namespace vibeqc::methods::generated",
            "// clang-format on",
            "",
            "#endif",
            "",
        ]
    )
    return "\n".join(lines)


def write_or_check(path: Path, text: str, *, check: bool) -> None:
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            raise SystemExit(
                f"stale generated method metadata: {path.relative_to(ROOT)}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    methods = load_manifest()
    write_or_check(C_IDS, emit_c_ids(methods), check=args.check)
    write_or_check(PYTHON, emit_python(methods), check=args.check)
    write_or_check(CPP, emit_cpp(methods), check=args.check)


if __name__ == "__main__":
    main()
