"""Synchronize VibeQC dispersion parameter catalogs from pinned upstream snapshots.

This is a development-time generator. Production code consumes the generated
method_parameters.json / _generated_parameters.py constants and has no runtime
dependency on simple-dftd3 or dftd4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "tools/parameters/dispersion_parameter_sources.json"
OVERRIDES = ROOT / "tools/parameters/method_parameter_overrides.json"
DEFAULT_OUTPUT = ROOT / "python/vibeqc_compiler/method/method_parameters.json"

_SYMBOL = re.compile(r"k[A-Z][A-Za-z0-9]*")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        return json.loads(raw)
    if raw == "true":
        return True
    if raw == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError as error:
            raise ValueError(f"unsupported inline TOML value {raw!r}") from error


def _parse_inline_table(line: str) -> dict[str, Any]:
    try:
        body = line[line.index("{") + 1 : line.rindex("}")]
    except ValueError as error:
        raise ValueError(f"expected inline TOML table: {line!r}") from error
    fields: list[str] = []
    start = 0
    quoted = False
    escaped = False
    for index, char in enumerate(body):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quoted:
            escaped = True
            continue
        if char == '"':
            quoted = not quoted
        elif char == "," and not quoted:
            fields.append(body[start:index])
            start = index + 1
    fields.append(body[start:])
    result: dict[str, Any] = {}
    for field in fields:
        if not field.strip():
            continue
        if "=" not in field:
            raise ValueError(f"invalid inline TOML field {field!r}")
        key, raw = field.split("=", 1)
        result[key.strip()] = _parse_value(raw)
    return result


def _method_key(raw: str) -> str:
    """Decode one TOML method key without interpreting dots inside quotes."""
    if raw.startswith('"') and raw.endswith('"'):
        return json.loads(raw)
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", raw):
        raise ValueError(f"unsupported TOML method key {raw!r}")
    return raw


def _parse_variant(path: Path, variant: str) -> dict[str, dict[str, Any]]:
    section = ""
    defaults: dict[str, Any] = {}
    records: dict[str, dict[str, Any]] = {}
    prefix = variant + " ="
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        match = re.fullmatch(r"\[([^]]+)\]", line)
        if match:
            section = match.group(1)
            continue
        if section == "default.parameter" and line.startswith(prefix):
            defaults = _parse_inline_table(line)
            continue
        method = re.fullmatch(r"parameter\.(.+)", section)
        if method and line.startswith(prefix):
            name = _method_key(method.group(1))
            if name in records:
                raise ValueError(f"duplicate parameter record {name!r}")
            records[name] = {**defaults, **_parse_inline_table(line)}
    if not defaults or not records:
        raise ValueError(f"{path} does not contain {variant} defaults and records")
    return records


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _display_name(upstream_key: str) -> str:
    return upstream_key.upper().replace("_", "-")


def _cpp_symbol(upstream_key: str, suffix: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", upstream_key)
    body = "".join(part[:1].upper() + part[1:] for part in parts)
    if not body or not body[0].isalpha():
        body = "P" + body
    symbol = "k" + body + suffix
    if not _SYMBOL.fullmatch(symbol):
        raise ValueError(f"cannot form C++ symbol from {upstream_key!r}")
    return symbol


def _verified_source(
    manifest: dict[str, Any], name: str
) -> tuple[Path, dict[str, Any]]:
    entry = manifest.get(name)
    if not isinstance(entry, dict):
        raise TypeError(f"missing source manifest entry {name!r}")
    snapshot = ROOT / str(entry.get("snapshot", ""))
    expected = entry.get("sha256")
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)
    if not isinstance(expected, str) or _sha256(snapshot) != expected:
        raise ValueError(f"pinned snapshot identity mismatch for {name}")
    return snapshot, entry


def _provenance(
    source: dict[str, Any],
    upstream_key: str,
    variant: str,
    *,
    doi: Any = None,
    projection: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "source": source["repository"],
        "revision": source["revision"],
        "path": source["path"],
        "sha256": source["sha256"],
        "upstream_key": upstream_key,
        "variant": variant,
    }
    if projection is not None:
        value["projection"] = projection
    if isinstance(doi, str) and doi:
        value["doi"] = doi
    return value


def build_catalog(
    source_manifest: Path = SOURCE_MANIFEST,
    overrides_path: Path = OVERRIDES,
) -> dict[str, Any]:
    manifest = _load_json(source_manifest)
    overrides = _load_json(overrides_path)
    if manifest.get("schema_version") != 1 or overrides.get("schema_version") != 1:
        raise ValueError("unsupported dispersion parameter source schema")

    d3_path, d3_source = _verified_source(manifest, "simple_dftd3")
    d4_path, d4_source = _verified_source(manifest, "dftd4")
    d3_upstream = _parse_variant(d3_path, "d3.bj")
    d4_upstream = _parse_variant(d4_path, "d4.bj-eeq-atm")

    d3_records: list[dict[str, Any]] = []
    for key in sorted(d3_upstream):
        source = d3_upstream[key]
        parameters = {
            "s6": _finite(source.get("s6"), f"d3/{key}.s6"),
            "s8": _finite(source.get("s8"), f"d3/{key}.s8"),
            "a1": _finite(source.get("a1"), f"d3/{key}.a1"),
            "a2": _finite(source.get("a2"), f"d3/{key}.a2"),
            # Production D3 is currently the explicitly two-body BJ slice.
            # Preserve upstream pair parameters while refusing to imply ATM.
            "s9": 0.0,
        }
        d3_records.append(
            {
                "name": f"{_display_name(key)}-D3(BJ)",
                "cpp_symbol": _cpp_symbol(key, "D3BJ"),
                "parameters": parameters,
                "provenance": _provenance(
                    d3_source,
                    key,
                    "d3.bj",
                    doi=source.get("doi"),
                    projection="two-body-s9=0",
                ),
            }
        )

    runtime = overrides.get("d4_runtime_defaults")
    profile_overrides = overrides.get("d4_profile_overrides")
    if not isinstance(runtime, dict) or not isinstance(profile_overrides, dict):
        raise TypeError("missing D4 runtime/profile overrides")
    d4_records = list(overrides.get("local_d4", []))
    for key in sorted(d4_upstream):
        source = d4_upstream[key]
        special = profile_overrides.get(key, {})
        if not isinstance(special, dict):
            raise TypeError(f"d4 profile override for {key!r} must be an object")
        resolved = {**runtime, **special}
        parameters = {
            "s6": _finite(source.get("s6"), f"d4/{key}.s6"),
            "s8": _finite(source.get("s8"), f"d4/{key}.s8"),
            "s9": _finite(source.get("s9"), f"d4/{key}.s9"),
            "a1": _finite(source.get("a1"), f"d4/{key}.a1"),
            "a2": _finite(source.get("a2"), f"d4/{key}.a2"),
            "ga": _finite(resolved.get("ga"), f"d4/{key}.ga"),
            "gc": _finite(resolved.get("gc"), f"d4/{key}.gc"),
            "profile": resolved["profile"],
            "reference_model": resolved["reference_model"],
            "charge_model": resolved["charge_model"],
            "cn_cutoff": _finite(resolved.get("cn_cutoff"), f"d4/{key}.cn_cutoff"),
            "pair_cutoff": _finite(
                resolved.get("pair_cutoff"), f"d4/{key}.pair_cutoff"
            ),
            "atm_cutoff": _finite(resolved.get("atm_cutoff"), f"d4/{key}.atm_cutoff"),
            "charge_cn_cutoff": _finite(
                resolved.get("charge_cn_cutoff"), f"d4/{key}.charge_cn_cutoff"
            ),
            "table_sha256": resolved["table_sha256"],
            "charge_parameter_sha256": resolved["charge_parameter_sha256"],
        }
        d4_records.append(
            {
                "name": special.get("name", f"{_display_name(key)}-D4(BJ-EEQ-ATM)"),
                "cpp_symbol": _cpp_symbol(key, "D4"),
                "python_spec": True,
                "parameters": parameters,
                "provenance": _provenance(
                    d4_source, key, "d4.bj-eeq-atm", doi=source.get("doi")
                ),
            }
        )

    payload = {
        "schema_version": 1,
        "d3_data_identity": overrides["d3_data_identity"],
        "d3_bj": d3_records,
        "d4": d4_records,
        "gcp": overrides["gcp"],
    }
    symbols: set[str] = set()
    for category in ("d3_bj", "d4", "gcp"):
        for record in payload[category]:
            symbol = record["cpp_symbol"]
            if symbol in symbols:
                raise ValueError(f"duplicate generated C++ symbol {symbol}")
            symbols.add(symbol)
    return payload


def _canonical_json_numbers(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_canonical_json_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonical_json_numbers(item) for key, item in value.items()}
    return value


def render_catalog(
    source_manifest: Path = SOURCE_MANIFEST,
    overrides_path: Path = OVERRIDES,
) -> str:
    payload = _canonical_json_numbers(build_catalog(source_manifest, overrides_path))
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--overrides", type=Path, default=OVERRIDES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(render_catalog(args.source_manifest, args.overrides))


if __name__ == "__main__":
    main()
