"""Reproduce maintained CUDA ownership separately from generated build output.

The ownership shards form a reviewed semantic classification, not a keyword classifier for
scientific mathematics. Exact source anchors partition mixed files, while the
inventory check rejects new or removed CUDA files until the ledger is updated.
Counts include host launch/ownership code in CUDA translation units. Generated
headers and translation units are measured only in an explicit build directory.
"""

import argparse
import hashlib
import json
import re
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("runtime", "scientific", "oracle", "fallback", "performance_exception")
SCIENTIFIC = set(ROLES) - {"runtime"}
CUDA_HEADER = re.compile(r"__global__|__device__|cuda_runtime(?:_api)?\.h")


def load_ledger(path: Path) -> dict[str, typing.Any]:
    """Load the sharded current ledger or a legacy monolithic JSON ledger."""
    if path.is_file():
        return json.loads(path.read_text())
    if not path.is_dir():
        raise ValueError(f"CUDA ownership ledger does not exist: {path}")

    meta_path = path / "meta.json"
    if not meta_path.is_file():
        raise ValueError(f"CUDA ownership shard directory lacks {meta_path.name}")
    ledger = json.loads(meta_path.read_text())

    subsystem_dir = path / "subsystems"
    generated_dir = path / "generated"
    file_dir = path / "files"
    for required in (subsystem_dir, generated_dir, file_dir):
        if not required.is_dir():
            raise ValueError(f"CUDA ownership shard directory lacks {required.name}/")

    subsystems: dict[str, typing.Any] = {}
    for shard in sorted(subsystem_dir.glob("*.json")):
        name = shard.stem
        if name in subsystems:
            raise ValueError(f"duplicate CUDA ownership subsystem shard: {name}")
        subsystems[name] = json.loads(shard.read_text())

    generated_families = []
    for shard in sorted(generated_dir.glob("*.json")):
        family = json.loads(shard.read_text())
        if family.get("name") != shard.stem:
            raise ValueError(
                f"generated-family shard name mismatch: {shard.name} "
                f"declares {family.get('name')!r}"
            )
        generated_families.append(family)

    files = [
        json.loads(shard.read_text()) for shard in sorted(file_dir.rglob("*.json"))
    ]
    return {
        **ledger,
        "generated_families": generated_families,
        "subsystems": subsystems,
        "files": files,
    }


def code_lines(source: typing.Any) -> typing.Any:
    """Count nonblank physical code lines, excluding C/C++ comments.

    Preserve quoted literals so a diagnostic containing // is still code.
    This is a counting lexer, not a parser for scientific function ownership.
    The semantic ledger owns that decision explicitly.
    """
    result, current = [], []
    block, quote, escaped, raw_end = False, None, False, None
    i = 0
    while i < len(source):
        char = source[i]
        if char == "\n":
            result.append(bool("".join(current).strip()))
            current = []
            escaped = False
            i += 1
            continue
        if block:
            if source.startswith("*/", i):
                block, i = False, i + 2
            else:
                i += 1
            continue
        if raw_end:
            if source.startswith(raw_end, i):
                current.extend(raw_end)
                i += len(raw_end)
                raw_end = None
            else:
                current.append(char)
                i += 1
            continue
        if quote:
            current.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            i += 1
            continue
        raw = (
            re.match(r'R"([^ ()\\\t\r\n]{0,16})\(', source[i : i + 20])
            if source.startswith('R"', i)
            else None
        )
        if raw:
            current.extend(raw[0])
            i += len(raw[0])
            raw_end = ")" + raw[1] + '"'
        elif source.startswith("//", i):
            end = source.find("\n", i)
            i = len(source) if end < 0 else end
        elif source.startswith("/*", i):
            block, i = True, i + 2
        else:
            current.append(char)
            # C++ digit separators are not character literals, including
            # hexadecimal forms such as 0xA'B'C. Encoding prefixes L/u8 on
            # actual character literals do not start with a numeric token.
            digit_separator = False
            if char == "'":
                begin = i - 1
                while begin >= 0 and (source[begin].isalnum() or source[begin] in "_'"):
                    begin -= 1
                digit_separator = (
                    begin + 1 < i
                    and source[begin + 1].isdigit()
                    and i + 1 < len(source)
                    and source[i + 1].isalnum()
                )
            if char in ('"', "'") and not digit_separator:
                quote = char
            i += 1
    if current or (source and not source.endswith("\n")):
        result.append(bool("".join(current).strip()))
    return result


def cuda_files(root: typing.Any) -> typing.Any:
    """Inventory source CUDA and CUDA-bearing shared C++ header fragments."""
    result = set()
    for path in (root / "src").rglob("*"):
        if path.is_file() and (
            path.suffix in (".cu", ".cuh")
            or (path.suffix == ".hpp" and CUDA_HEADER.search(path.read_text()))
        ):
            result.add(path.relative_to(root).as_posix())
    return result


def _anchor(lines: typing.Any, text: typing.Any) -> typing.Any:
    if text is None:
        return len(lines)
    found = [i for i, line in enumerate(lines) if text in line]
    if len(found) != 1:
        raise ValueError(f"ownership anchor must occur exactly once: {text!r}")
    return found[0]


def ownership_report(
    root: typing.Any, ledger: typing.Any, build: typing.Any = None
) -> typing.Any:
    """Check complete source coverage and return stable per-region/file totals."""
    if ledger.get("schema") != "vibeqc.cuda-ownership.v1":
        raise ValueError("unsupported CUDA ownership ledger")
    records = ledger["files"]
    paths = [row["path"] for row in records]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate CUDA ownership file")
    actual = cuda_files(root)
    if set(paths) != actual:
        raise ValueError(
            f"CUDA inventory differs: unclassified={sorted(actual - set(paths))}; stale={sorted(set(paths) - actual)}"
        )
    owners = ledger["subsystems"]
    for name, owner in owners.items():
        for field in (
            "owner",
            "current_default",
            "generated_capability",
            "missing_capability",
            "retirement_condition",
            "evidence",
            "status",
        ):
            if not owner.get(field):
                raise ValueError(f"subsystem {name} lacks {field}")
        evidence = owner["evidence"]
        if not isinstance(evidence, list) or not all(
            isinstance(path, str) and (root / path).is_file() for path in evidence
        ):
            raise ValueError(f"subsystem {name} has stale evidence paths")
    files, totals, subsystems = [], dict.fromkeys(ROLES, 0), {}
    for row in sorted(records, key=lambda value: value["path"]):
        source = (root / row["path"]).read_text()
        lines, active = source.splitlines(), code_lines(source)
        if len(lines) != len(active):
            raise ValueError("counting lexer lost a physical source line")
        if (
            row["subsystem"] not in owners
            or row["role"] not in ROLES
            or not row["reason"]
        ):
            raise ValueError(f"invalid semantic ownership for {row['path']}")
        roles = [row["role"]] * len(lines)
        occupied, regions = set(), []
        for region in row.get("regions", []):
            begin = _anchor(lines, region["start"])
            end = _anchor(lines, region.get("stop"))
            if begin >= end or region["role"] not in ROLES or not region["reason"]:
                raise ValueError(f"invalid ownership region in {row['path']}")
            indices = set(range(begin, end))
            if occupied & indices:
                raise ValueError(f"overlapping ownership regions in {row['path']}")
            occupied |= indices
            roles[begin:end] = [region["role"]] * (end - begin)
            regions.append(
                {
                    **region,
                    "first_line": begin + 1,
                    "last_line": end,
                    "code_lines": sum(active[begin:end]),
                }
            )
        counts = {
            role: sum(
                code and assigned == role
                for code, assigned in zip(active, roles, strict=True)
            )
            for role in ROLES
        }
        subsystem = subsystems.setdefault(row["subsystem"], dict.fromkeys(ROLES, 0))
        for role in ROLES:
            totals[role] += counts[role]
            subsystem[role] += counts[role]
        files.append(
            {
                **row,
                "regions": regions,
                "physical_lines": len(lines),
                "code_lines": counts,
                "bytes": len(source.encode()),
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
            }
        )
    generated, seen = [], {}
    names = [family["name"] for family in ledger["generated_families"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate generated family name")
    for family in ledger["generated_families"]:
        owner = family.get("owner")
        if not isinstance(owner, str) or not (root / owner).exists():
            raise ValueError(f"generated family {family['name']} has stale owner path")
        found = []
        if build is not None:
            for pattern in family["outputs"]:
                for path in sorted(build.glob(pattern)):
                    if path.is_file():
                        if path in seen:
                            if seen[path] != family["name"]:
                                raise ValueError(
                                    f"generated output claimed by multiple families: {path}"
                                )
                            continue
                        seen[path] = family["name"]
                        raw = path.read_bytes()
                        found.append(
                            {
                                "path": path.relative_to(build).as_posix(),
                                "bytes": len(raw),
                                "code_lines": sum(code_lines(raw.decode())),
                                "sha256": hashlib.sha256(raw).hexdigest(),
                            }
                        )
        generated.append(
            {
                **family,
                "measurement": "explicit build"
                if build is not None
                else "not materialized",
                "files": found,
            }
        )
    return {
        "schema": "vibeqc.cuda-ownership-report.v1",
        "maintained_code_lines": totals,
        # Reclassifying scientific code as an oracle/exception cannot be
        # advertised as deleting handwritten scientific implementation.
        "all_handwritten_scientific_lines": sum(totals[r] for r in SCIENTIFIC),
        "per_subsystem": subsystems,
        "files": files,
        "migration_ledger": owners,
        "generated": generated,
        "generated_bytes": sum(f["bytes"] for g in generated for f in g["files"]),
        "counting_scope": "nonblank noncomment physical lines in source CUDA translation units/headers; includes host launch code; explicit source regions own semantics",
    }


def validate_baseline(baseline: typing.Any) -> None:
    """Reject internally inconsistent historical totals before claiming a delta."""
    if baseline.get("schema") != "vibeqc.cuda-ownership-report.v1":
        raise ValueError("baseline uses a different ownership report schema")
    totals = dict.fromkeys(ROLES, 0)
    paths = set()
    for row in baseline["files"]:
        if row["path"] in paths:
            raise ValueError("duplicate baseline file")
        paths.add(row["path"])
        for role in ROLES:
            value = row["code_lines"][role]
            if type(value) is not int or value < 0:
                raise ValueError("invalid baseline code-line count")
            totals[role] += value
    if (
        totals != baseline["maintained_code_lines"]
        or sum(totals[role] for role in SCIENTIFIC)
        != baseline["all_handwritten_scientific_lines"]
    ):
        raise ValueError("baseline totals differ from file records")
    generated_bytes = sum(
        file["bytes"] for family in baseline["generated"] for file in family["files"]
    )
    if baseline["generated_bytes"] != generated_bytes:
        raise ValueError("baseline generated byte total differs from file records")


def main() -> typing.Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=ROOT / "docs/cuda_ownership")
    parser.add_argument("--build", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = ownership_report(ROOT, load_ledger(args.ledger), args.build)
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        validate_baseline(baseline)
        old = {row["path"]: row for row in baseline["files"]}
        current = {row["path"]: row for row in report["files"]}
        report["delta"] = {
            "maintained_code_lines": {
                role: report["maintained_code_lines"][role]
                - baseline["maintained_code_lines"][role]
                for role in ROLES
            },
            "added_files": sorted(current.keys() - old.keys()),
            "removed_files": sorted(old.keys() - current.keys()),
            # A classification change is visible even when a source edit also
            # occurs. Reviewers must distinguish it from physical retirement.
            "reclassified_files": sorted(
                path
                for path in old.keys() & current.keys()
                if old[path]["role"] != current[path]["role"]
                or [
                    (r["start"], r.get("stop"), r["role"]) for r in old[path]["regions"]
                ]
                != [
                    (r["start"], r.get("stop"), r["role"])
                    for r in current[path]["regions"]
                ]
            ),
        }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.check:
        print(f"CUDA ownership inventory checked: {len(report['files'])} files")
    elif args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
