#!/usr/bin/env python3
"""Audit pinned Libxc Maple syntax and static Graph-lowering coverage."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.xc import libxc_maple

LIBXC_ROOT = ROOT / "upstream/libxc/7.0.0"

_INCLUDE = re.compile(r'^\s*\$include\s+"([^"]+)"\s*$', re.MULTILINE)
_CONDITIONAL = re.compile(r"^\s*\$(?:ifdef|elif)\s+([A-Za-z_]\w*)\s*$", re.MULTILINE)
_FUNCTIONAL_TYPE = re.compile(r"\(\*\s*type:\s*([^*]+?)\s*\*\)")


def _include_closure(entry: Path) -> tuple[Path, ...]:
    """Return the deterministic transitive include closure for one Maple entry."""
    seen: set[Path] = set()
    ordered: list[Path] = []

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        ordered.append(path)
        source = path.read_text(encoding="utf-8")
        for name in _INCLUDE.findall(source):
            target = path.parent / name
            if target.is_file():
                visit(target)

    visit(entry)
    return tuple(ordered)


def _profiles(entry: Path) -> tuple[tuple[str, ...], ...]:
    """Exercise baseline plus each externally selectable conditional arm."""
    symbols = sorted(
        {
            symbol
            for source in _include_closure(entry)
            for symbol in _CONDITIONAL.findall(source.read_text(encoding="utf-8"))
        }
    )
    return ((), *((symbol,) for symbol in symbols))


def _functional_type(source: str) -> str | None:
    match = _FUNCTIONAL_TYPE.search(source)
    return match.group(1).strip() if match else None


def _definition_expression(
    module: libxc_maple.MapleModule, kind: str, name: str
) -> tuple[str, tuple[str, ...]]:
    if kind == "function":
        function = dict(module.functions)[name]
        return function.expression, function.parameters
    return dict(module.assignments)[name], ()


def _expression_nodes(expression: str) -> tuple[ast.AST, ...]:
    """Include deferred bounded-sum terms in the static expression graph."""
    pending = [expression]
    seen: set[str] = set()
    nodes: list[ast.AST] = []
    while pending:
        text = pending.pop()
        if text in seen:
            continue
        seen.add(text)
        tree = libxc_maple._parse_expression(text)
        for node in ast.walk(tree):
            nodes.append(node)
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_maple_bounded_add"
            ):
                continue
            if len(node.args) != 3 or not isinstance(node.args[2], ast.List):
                raise libxc_maple.MapleImportError("invalid encoded bounded sum")
            for term in node.args[2].elts:
                if not isinstance(term, ast.Constant) or not isinstance(
                    term.value, str
                ):
                    raise libxc_maple.MapleImportError(
                        "invalid encoded bounded-sum term"
                    )
                pending.append(term.value)
    return tuple(nodes)


def _entry_blockers(
    module: libxc_maple.MapleModule, *, require_entry: bool = True
) -> tuple[str, ...]:
    """Find unsupported callable targets reachable from the public Maple f."""
    functions = dict(module.functions)
    assignments = dict(module.assignments)
    if "f" not in functions:
        return ("missing callable entry f",) if require_entry else ()

    evaluator = libxc_maple._Evaluator(module, Graph())
    pending: list[tuple[str, str]] = [("function", "f")]
    visited: set[tuple[str, str]] = set()
    blockers: set[str] = set()

    while pending:
        kind, name = pending.pop()
        key = (kind, name)
        if key in visited:
            continue
        visited.add(key)
        expression, parameters = _definition_expression(module, kind, name)
        nodes = _expression_nodes(expression)
        local_names = set(parameters)

        references = {
            node.id
            for node in nodes
            if isinstance(node, ast.Name) and node.id not in local_names
        }
        for reference in references:
            if reference in functions:
                pending.append(("function", reference))
            elif reference in assignments:
                pending.append(("assignment", reference))

        for node in nodes:
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            target = node.func.id
            if target in local_names or target in functions:
                continue
            if target == "_maple_bounded_add":
                continue

            if target in assignments:
                blockers.add(target)
                continue
            try:
                resolved = evaluator._name(target, {})
            except libxc_maple.MapleImportError:
                blockers.add(target)
                continue
            if not isinstance(
                resolved, (libxc_maple._IntrinsicRef, libxc_maple._FunctionRef)
            ):
                blockers.add(target)

    return tuple(sorted(blockers))


def audit(root: Path = LIBXC_ROOT) -> dict[str, typing.Any]:
    """Audit every pinned .mpl source without changing production semantics."""
    if not root.is_dir():
        raise ValueError("Maple audit root must be an existing directory")
    entries = sorted(path for path in root.glob("*.mpl") if path.is_file())
    if not entries:
        raise ValueError("Maple audit root contains no .mpl sources")
    sources: list[dict[str, typing.Any]] = []
    for entry in entries:
        text = entry.read_text(encoding="utf-8")
        functional_type = _functional_type(text)
        is_functional = functional_type is not None
        profiles: list[dict[str, typing.Any]] = []
        blockers: set[str] = set()

        for defines in _profiles(entry):
            record: dict[str, typing.Any] = {"defines": list(defines)}
            try:
                module = libxc_maple.import_maple_file(
                    root,
                    entry.name,
                    defines=defines,
                    allow_duplicate_includes=True,
                )
                entry_blockers = _entry_blockers(module, require_entry=is_functional)
            except libxc_maple.MapleImportError as error:
                record["error"] = str(error)
            else:
                record["functions"] = len(module.functions)
                record["assignments"] = len(module.assignments)
                record["entry_blockers"] = list(entry_blockers)
                blockers.update(entry_blockers)
            profiles.append(record)

        parse_errors = [item for item in profiles if "error" in item]
        sources.append(
            {
                "file": entry.name,
                "kind": "functional" if is_functional else "support",
                "functional_type": functional_type,
                "profiles": profiles,
                "parse_profiles": len(profiles),
                "parse_failures": len(parse_errors),
                "entry_blockers": sorted(blockers),
                "direct_importable": bool(
                    is_functional and not parse_errors and not blockers
                ),
            }
        )

    functional = [item for item in sources if item["kind"] == "functional"]
    support = [item for item in sources if item["kind"] == "support"]
    direct = [item for item in functional if item["direct_importable"]]
    blocked = [item for item in functional if not item["direct_importable"]]
    return {
        "importer_semantics": libxc_maple.IMPORTER_SEMANTICS,
        "root": root.as_posix(),
        "source_files": len(sources),
        "functional_files": len(functional),
        "support_files": len(support),
        "direct_importable_functionals": len(direct),
        "blocked_functionals": len(blocked),
        "sources": sources,
    }


def render_markdown(report: dict[str, typing.Any]) -> str:
    """Render a reviewable checked-in snapshot of the audit."""
    lines = [
        "# Libxc Maple importer coverage",
        "",
        "<!-- Generated by tools/audit_libxc_maple_coverage.py --markdown. -->",
        "",
        f"- Importer semantics: {report['importer_semantics']}",
        (
            f"- Pinned Maple files: **{report['source_files']}** "
            f"({report['functional_files']} functional + {report['support_files']} support)"
        ),
        (
            f"- Static direct-import candidates: "
            f"**{report['direct_importable_functionals']}/{report['functional_files']}**"
        ),
        "",
        (
            "A direct-import candidate means every audited conditional profile parses and "
            "the top-level Maple f call graph contains no callable target unknown to "
            "the current Graph evaluator. It is not a production-support claim: numerical "
            "qualification, bindings, endpoint policy, and CPU/CUDA gates remain separate."
        ),
        "",
        "| Source | Type | Profiles | Parse failures | Entry blockers | Direct candidate |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for item in report["sources"]:
        blockers = ", ".join(item["entry_blockers"]) or "—"
        functional_type = item["functional_type"] or "support"
        direct = "yes" if item["direct_importable"] else "no"
        lines.append(
            f"| {item['file']} | {functional_type} | {item['parse_profiles']} | "
            f"{item['parse_failures']} | {blockers} | {direct} |"
        )

    support_failures = [
        item
        for item in report["sources"]
        if item["kind"] == "support" and item["parse_failures"]
    ]

    lines.extend(["", "## Current evaluator gaps", ""])
    functional_failures = [
        item
        for item in report["sources"]
        if item["kind"] == "functional" and item["parse_failures"]
    ]
    if functional_failures:
        for item in functional_failures:
            lines.append(
                f"- {item['file']} fails parsing in {item['parse_failures']}/"
                f"{item['parse_profiles']} audited profiles."
            )
    else:
        lines.append(
            "- No functional source has a parser-syntax failure in the audited profiles."
        )
    for item in report["sources"]:
        if item["kind"] != "functional":
            continue
        for blocker in item["entry_blockers"]:
            if blocker == "missing callable entry f":
                lines.append(
                    f"- {item['file']} lacks a callable Maple f entry in at least one audited profile."
                )
            else:
                lines.append(
                    f"- {item['file']} reaches {blocker}, which is not exposed as a callable "
                    "Maple target by the current Graph evaluator."
                )
    for item in support_failures:
        lines.append(
            f"- {item['file']} has {item['parse_failures']}/{item['parse_profiles']} "
            "standalone support-import failures. Functional imports retain their "
            "separate evaluator/include context."
        )
    lines.extend(
        [
            "",
            (
                "These are importer/evaluator coverage findings only. They do not weaken the "
                "existing requirement for independent E/vxc/fxc, endpoint, provenance, and "
                "backend qualification before a functional is cut over to production."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--json", action="store_true")
    mode.add_argument("--markdown", action="store_true")
    parser.add_argument("--root", type=Path, default=LIBXC_ROOT)
    args = parser.parse_args()

    try:
        report = audit(args.root)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.markdown:
        print(render_markdown(report), end="")
    else:
        print(
            f"{report['importer_semantics']}: {report['source_files']} Maple files; "
            f"{report['direct_importable_functionals']}/{report['functional_files']} "
            "functional entries are static direct-import candidates"
        )
        for item in report["sources"]:
            if item["kind"] == "functional" and not item["direct_importable"]:
                reason = item["entry_blockers"] or [
                    f"{item['parse_failures']} parse profile failure(s)"
                ]
                print(f"- {item['file']}: {', '.join(reason)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
