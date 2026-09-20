"""Generate runtime-free Python and C++ method-parameter constants."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "python/vibeqc_compiler/method/method_parameters.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CPP_SYMBOL = re.compile(r"k[A-Z][A-Za-z0-9]*")

_D3_FIELDS = ("s6", "s8", "a1", "a2", "s9")
_D4_CPP_FIELDS = (
    "s6",
    "s8",
    "s9",
    "a1",
    "a2",
    "cn_cutoff",
    "pair_cutoff",
    "atm_cutoff",
    "ga",
    "gc",
)
_GCP_CPP_FIELDS = ("sigma", "alpha", "beta", "damping_scale", "damping_exponent")


_PYTHON_PARAMETER_TYPES = {
    "d3_bj": {
        "s6": "float",
        "s8": "float",
        "a1": "float",
        "a2": "float",
        "s9": "float",
        "table_sha256": "str",
        "radii_sha256": "str",
    },
    "d4": {
        "s6": "float",
        "s8": "float",
        "s9": "float",
        "a1": "float",
        "a2": "float",
        "ga": "float",
        "gc": "float",
        "profile": "str",
        "reference_model": "str",
        "charge_model": "str",
        "cn_cutoff": "float",
        "pair_cutoff": "float",
        "atm_cutoff": "float",
        "charge_cn_cutoff": "float",
        "table_sha256": "str",
        "charge_parameter_sha256": "str",
    },
    "gcp": {
        "basis": "str",
        "sigma": "float",
        "eta": "float",
        "eta_spec": "float",
        "alpha": "float",
        "beta": "float",
        "damping_scale": "float",
        "damping_exponent": "float",
        "damping": "bool",
        "parameter_sha256": "str",
        "implementation_sha256": "str",
        "vdw_radii_sha256": "str",
        "data_sha256": "str",
        "source": "str",
        "source_revision": "str",
        "license": "str",
        "profile": "str",
        "supported_atomic_numbers": "ints",
    },
}


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _validate_record(
    record: dict[str, Any], category: str, fields: tuple[str, ...]
) -> None:
    name = record.get("name")
    symbol = record.get("cpp_symbol")
    parameters = record.get("parameters")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{category} record requires a non-empty name")
    if not isinstance(symbol, str) or not _CPP_SYMBOL.fullmatch(symbol):
        raise ValueError(f"{category}/{name} has an invalid C++ symbol")
    if not isinstance(parameters, dict):
        raise TypeError(f"{category}/{name} requires a parameter object")
    for field in fields:
        _finite_number(parameters.get(field), f"{category}/{name}.{field}")


def load_source(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported method-parameter schema")
    identity = payload.get("d3_data_identity")
    if not isinstance(identity, dict):
        raise TypeError("missing D3 data identity")
    for key in ("table_sha256", "radii_sha256"):
        value = identity.get(key)
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(
                f"d3_data_identity.{key} must be a lowercase SHA-256 digest"
            )

    symbols: set[str] = set()
    names: set[tuple[str, str]] = set()
    for category, fields in (
        ("d3_bj", _D3_FIELDS),
        ("d4", _D4_CPP_FIELDS),
        ("gcp", _GCP_CPP_FIELDS),
    ):
        records = payload.get(category)
        if not isinstance(records, list) or not records:
            raise ValueError(f"{category} must contain at least one record")
        for record in records:
            if not isinstance(record, dict):
                raise TypeError(f"{category} entries must be objects")
            _validate_record(record, category, fields)
            key = (category, record["name"])
            if key in names:
                raise ValueError(
                    f"duplicate parameter record {category}/{record['name']}"
                )
            if record["cpp_symbol"] in symbols:
                raise ValueError(f"duplicate C++ symbol {record['cpp_symbol']}")
            names.add(key)
            symbols.add(record["cpp_symbol"])

    for record in payload["d4"]:
        params = record["parameters"]
        if record.get("python_spec"):
            for key in ("table_sha256", "charge_parameter_sha256"):
                value = params.get(key)
                if not isinstance(value, str) or not _SHA256.fullmatch(value):
                    raise ValueError(
                        f"d4/{record['name']}.{key} must be a SHA-256 digest"
                    )

    for record in payload["gcp"]:
        params = record["parameters"]
        for key in (
            "parameter_sha256",
            "implementation_sha256",
            "vdw_radii_sha256",
            "data_sha256",
        ):
            value = params.get(key)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"gcp/{record['name']}.{key} must be a SHA-256 digest")
        numbers = params.get("supported_atomic_numbers")
        if (
            not isinstance(numbers, list)
            or not numbers
            or any(
                isinstance(number, bool)
                or not isinstance(number, int)
                or not 1 <= number <= 118
                for number in numbers
            )
            or numbers != sorted(set(numbers))
        ):
            raise ValueError(
                "gCP supported_atomic_numbers must be sorted, unique integers in [1, 118]"
            )
    for category in ("d3_bj", "d4", "gcp"):
        for record in payload[category]:
            if category == "d4" and not record.get("python_spec"):
                continue
            params = dict(record["parameters"])
            if category == "d3_bj":
                params.update(identity)
            _validate_python_parameters(params, category)
    return payload, hashlib.sha256(raw).hexdigest()


def _py_value(value: Any) -> str:
    if isinstance(value, list):
        return (
            "("
            + ", ".join(repr(item) for item in value)
            + ("," if len(value) == 1 else "")
            + ")"
        )
    return repr(value)


def _render_py_mapping(
    name: str, records: list[tuple[str, dict[str, Any]]]
) -> list[str]:
    lines = [f"{name} = MappingProxyType({{"]
    for record_name, parameters in records:
        lines.append(f"    {record_name!r}: MappingProxyType({{")
        for key, value in parameters.items():
            lines.append(f"        {key!r}: {_py_value(value)},")
        lines.append("    }),")
    lines.append("})")
    return lines


def _render_parameter_accessors() -> list[str]:
    lines = [
        "def _parameter_float(value: object) -> float:",
        "    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):",
        "        raise TypeError('parameter must be a finite real scalar')",
        "    return float(value)",
        "",
        "def _parameter_str(value: object) -> str:",
        "    if not isinstance(value, str) or not value:",
        "        raise TypeError('parameter must be a nonempty string')",
        "    return value",
        "",
        "def _parameter_bool(value: object) -> bool:",
        "    if type(value) is not bool:",
        "        raise TypeError('parameter must be a bool')",
        "    return value",
        "",
        "def _parameter_ints(value: object) -> tuple[int, ...]:",
        "    if not isinstance(value, tuple) or not value or any(type(x) is not int for x in value):",
        "        raise TypeError('parameter must be a nonempty integer tuple')",
        "    return tuple(int(x) for x in value)",
        "",
    ]
    for category, record, table, accessor in (
        ("d3_bj", "D3Parameters", "D3_BJ_PARAMETER_SETS", "d3_parameters"),
        ("d4", "D4Parameters", "D4_PARAMETER_SETS", "d4_parameters"),
        ("gcp", "GCPParameters", "GCP_PARAMETER_SETS", "gcp_parameters"),
    ):
        schema = _PYTHON_PARAMETER_TYPES[category]
        lines += [f"class {record}(TypedDict):"]
        for field, kind in schema.items():
            annotation = "tuple[int, ...]" if kind == "ints" else kind
            lines.append(f"    {field}: {annotation}")
        lines += [
            "",
            f"def {accessor}(name: str) -> {record}:",
            f"    value = {table}[name]",
            "    return {",
        ]
        for field, kind in schema.items():
            lines.append(f"        {field!r}: _parameter_{kind}(value[{field!r}]),")
        lines += ["    }", ""]
    return lines


def _validate_python_parameters(parameters: dict[str, Any], category: str) -> None:
    schema = _PYTHON_PARAMETER_TYPES[category]
    if set(parameters) != set(schema):
        raise ValueError(f"{category} parameter fields differ from the typed schema")
    for name, kind in schema.items():
        value = parameters[name]
        if kind == "float":
            _finite_number(value, f"{category}.{name}")
        elif kind == "str" and (not isinstance(value, str) or not value):
            raise TypeError(f"{category}.{name} must be a nonempty string")
        elif kind == "bool" and type(value) is not bool:
            raise TypeError(f"{category}.{name} must be a bool")
        elif kind == "ints" and (
            not isinstance(value, list)
            or not value
            or any(type(item) is not int for item in value)
        ):
            raise TypeError(f"{category}.{name} must be integer values")


def render_python(payload: dict[str, Any], source_sha256: str) -> str:
    identity = payload["d3_data_identity"]
    d3_records: list[tuple[str, dict[str, Any]]] = []
    for record in payload["d3_bj"]:
        params = dict(record["parameters"])
        params["table_sha256"] = identity["table_sha256"]
        params["radii_sha256"] = identity["radii_sha256"]
        d3_records.append((record["name"], params))

    d4_records = [
        (record["name"], dict(record["parameters"]))
        for record in payload["d4"]
        if record.get("python_spec")
    ]
    gcp_records = []
    for record in payload["gcp"]:
        params = dict(record["parameters"])
        params["supported_atomic_numbers"] = tuple(params["supported_atomic_numbers"])
        gcp_records.append((record["name"], params))

    provenance: dict[str, dict[str, Any]] = {}
    for category in ("d3_bj", "d4", "gcp"):
        for record in payload[category]:
            provenance[f"{category}:{record['name']}"] = dict(
                record.get("provenance", {})
            )

    lines = [
        '"""Generated from method_parameters.json; do not edit by hand."""',
        "",
        "import math",
        "from types import MappingProxyType",
        "from typing import TypedDict",
        "",
        f"PARAMETER_SOURCE_SHA256 = {source_sha256!r}",
        f"D3_TABLE_SHA256 = {identity['table_sha256']!r}",
        f"D3_RADII_SHA256 = {identity['radii_sha256']!r}",
        "",
    ]
    lines.extend(_render_py_mapping("D3_BJ_PARAMETER_SETS", d3_records))
    lines.append("")
    lines.extend(_render_py_mapping("D4_PARAMETER_SETS", d4_records))
    lines.append("")
    lines.extend(_render_py_mapping("GCP_PARAMETER_SETS", gcp_records))
    lines.append("")
    lines.extend(_render_py_mapping("PARAMETER_PROVENANCE", list(provenance.items())))
    lines.append("")
    lines.extend(_render_parameter_accessors())
    return "\n".join(lines)


def _cpp_float(value: Any) -> str:
    number = _finite_number(value, "C++ parameter")
    text = repr(number)
    if "e" not in text and "." not in text:
        text += ".0"
    return text


def _render_cpp_record(
    symbol: str, struct_name: str, params: dict[str, Any], fields: tuple[str, ...]
) -> str:
    values = ", ".join(_cpp_float(params[field]) for field in fields)
    accessor = symbol[1].lower() + symbol[2:]
    return (
        f"VIBEQC_METHOD_PARAMS_HD inline constexpr {struct_name} {accessor}() "
        f"{{ return {{{values}}}; }}"
    )


def render_cpp(payload: dict[str, Any], source_sha256: str) -> str:
    lines = [
        "#pragma once",
        "",
        "// Generated from python/vibeqc_compiler/method/method_parameters.json.",
        "// Do not edit by hand.",
        "",
        "#if defined(__CUDACC__)",
        "#define VIBEQC_METHOD_PARAMS_HD __host__ __device__",
        "#else",
        "#define VIBEQC_METHOD_PARAMS_HD",
        "#endif",
        "",
        "namespace vibeqc::generated::method_parameters {",
        "",
        f'inline constexpr char kParameterSourceSha256[] = "{source_sha256}";',
        "",
        "struct D3BJParameters {",
        "  double s6, s8, a1, a2, s9;",
        "};",
    ]
    for record in payload["d3_bj"]:
        lines.append(
            _render_cpp_record(
                record["cpp_symbol"], "D3BJParameters", record["parameters"], _D3_FIELDS
            )
        )
    lines.extend(
        [
            "",
            "struct D4Parameters {",
            "  double s6, s8, s9, a1, a2;",
            "  double cn_cutoff, pair_cutoff, atm_cutoff;",
            "  double ga, gc;",
            "};",
        ]
    )
    for record in payload["d4"]:
        lines.append(
            _render_cpp_record(
                record["cpp_symbol"],
                "D4Parameters",
                record["parameters"],
                _D4_CPP_FIELDS,
            )
        )
        params = record["parameters"]
        if "charge_cn_cutoff" in params:
            accessor = record["cpp_symbol"][1].lower() + record["cpp_symbol"][2:]
            cutoff = _cpp_float(params["charge_cn_cutoff"])
            lines.append(
                f"VIBEQC_METHOD_PARAMS_HD inline constexpr double {accessor}ChargeCnCutoff() "
                f"{{ return {cutoff}; }}"
            )
    lines.extend(
        [
            "",
            "struct GCPParameters {",
            "  double sigma, alpha, beta, damping_scale, damping_exponent;",
            "};",
        ]
    )
    for record in payload["gcp"]:
        lines.append(
            _render_cpp_record(
                record["cpp_symbol"],
                "GCPParameters",
                record["parameters"],
                _GCP_CPP_FIELDS,
            )
        )
        supported = record["parameters"]["supported_atomic_numbers"]
        accessor = record["cpp_symbol"][1].lower() + record["cpp_symbol"][2:]
        if supported == list(range(supported[0], supported[-1] + 1)):
            expression = f"z >= {supported[0]} && z <= {supported[-1]}"
        else:
            expression = " || ".join(f"z == {z}" for z in supported)
        lines.append(
            f"VIBEQC_METHOD_PARAMS_HD inline constexpr bool {accessor}SupportsAtomicNumber(int z) "
            f"{{ return {expression}; }}"
        )
    lines.extend(
        [
            "",
            "}  // namespace vibeqc::generated::method_parameters",
            "",
            "#undef VIBEQC_METHOD_PARAMS_HD",
            "",
        ]
    )
    return "\n".join(lines)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--python-output", type=Path)
    parser.add_argument("--cpp-output", type=Path)
    args = parser.parse_args()
    if args.python_output is None and args.cpp_output is None:
        parser.error("at least one output is required")

    payload, source_sha256 = load_source(args.source)
    if args.python_output is not None:
        _write(args.python_output, render_python(payload, source_sha256))
    if args.cpp_output is not None:
        _write(args.cpp_output, render_cpp(payload, source_sha256))


if __name__ == "__main__":
    main()
