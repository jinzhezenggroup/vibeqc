"""Enforce cross-method native ownership boundaries and report architecture metrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {".cpp", ".hpp", ".cu", ".cuh"}

# These layers are intended to remain reusable by every electronic-structure
# method. They may depend on one another and on chemistry/integral primitives,
# but never acquire new concrete HF/DFT/post-HF/CC implementation dependencies.
SHARED_OWNERS = ("core", "runtime", "tensor", "response")
METHOD_PREFIXES = ("scf/", "dft/", "posthf/", "cc/")

# Current reverse edges are explicit debt ceilings, not approved design. The
# check allows them to disappear but rejects any new shared -> method edge.
KNOWN_METHOD_EDGES = {
    ("runtime/cuda_runtime.cu", "scf/aot_shell_registry.hpp"),
    ("runtime/host_component_trace.hpp", "scf/reference/observation.hpp"),
}

# The current CC solver still owns local iteration infrastructure. Keep its
# present count as a ceiling while the shared iterative-solver refactor lands.
KNOWN_DUPLICATE_INFRASTRUCTURE = {
    "cc_cpu_diis_owner": 1,
    "cc_cpu_local_linear_solver": 1,
    "cc_cuda_diis_owner": 1,
}
DUPLICATE_PATTERNS = {
    "cc_cpu_diis_owner": re.compile(r"\bstruct\s+Diis\b"),
    "cc_cpu_local_linear_solver": re.compile(
        r"\bbool\s+solve_linear\s*\(\s*std::vector<double>"
    ),
    "cc_cuda_diis_owner": re.compile(r"\bvoid\s+run_diis\s*\("),
}

COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]', re.MULTILINE)


def _without_comments(text: str) -> str:
    """Remove comments while preserving line numbering."""

    def replacement(match: re.Match[str]) -> str:
        return " " + "\n" * match.group(0).count("\n")

    return COMMENT_RE.sub(replacement, text)


def _source_target(source: Path, path: Path, include: str) -> str | None:
    """Resolve a repository-local include to its src-relative spelling."""
    if include.startswith(("./", "../")):
        candidate = (path.parent / include).resolve()
    else:
        local = path.parent / include
        candidate = (local if local.exists() else source / include).resolve()
    try:
        relative = candidate.relative_to(source).as_posix()
    except ValueError:
        return None
    if candidate.exists() or relative.startswith(
        ("core/", "runtime/", "tensor/", "response/", *METHOD_PREFIXES)
    ):
        return relative
    return None


def _native_files(root: Path) -> list[Path]:
    source = root / "src"
    if not source.is_dir():
        return []
    return sorted(
        path for path in source.rglob("*") if path.is_file() and path.suffix in SUFFIXES
    )


def _area_metrics(root: Path) -> dict[str, dict[str, int]]:
    """Return maintainability inventory without imposing size thresholds."""
    source = root / "src"
    areas: dict[str, dict[str, int]] = {}
    for path in _native_files(root):
        relative = path.relative_to(source)
        area = relative.parts[0]
        text = path.read_text(encoding="utf-8")
        metrics = areas.setdefault(
            area,
            {"files": 0, "lines": 0, "bytes": 0, "generated_lines": 0},
        )
        metrics["files"] += 1
        metrics["lines"] += len(text.splitlines())
        metrics["bytes"] += path.stat().st_size
        if "generated" in path.name:
            metrics["generated_lines"] += len(text.splitlines())
    return dict(sorted(areas.items()))


def _duplicate_infrastructure(root: Path) -> dict[str, dict[str, object]]:
    source = root / "src"
    cc = source / "cc"
    report: dict[str, dict[str, object]] = {}
    for name, pattern in DUPLICATE_PATTERNS.items():
        matches: list[str] = []
        if cc.is_dir():
            for path in sorted(cc.rglob("*")):
                if not path.is_file() or path.suffix not in SUFFIXES:
                    continue
                text = _without_comments(path.read_text(encoding="utf-8"))
                for match in pattern.finditer(text):
                    line = text.count("\n", 0, match.start()) + 1
                    matches.append(f"{path.relative_to(source).as_posix()}:{line}")
        report[name] = {
            "count": len(matches),
            "allowed": KNOWN_DUPLICATE_INFRASTRUCTURE[name],
            "locations": matches,
        }
    return report


def audit_electronic_structure_boundaries(root: Path = ROOT) -> dict[str, object]:
    """Audit reusable native layers against concrete method dependencies."""
    source = (root / "src").resolve()
    errors: list[str] = []
    edges: list[dict[str, str]] = []
    method_edges: list[dict[str, object]] = []
    modules: list[dict[str, object]] = []

    for owner in SHARED_OWNERS:
        directory = source / owner
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix not in SUFFIXES:
                continue
            content = path.read_text(encoding="utf-8")
            relative = path.relative_to(source).as_posix()
            modules.append(
                {
                    "owner": owner,
                    "path": relative,
                    "lines": len(content.splitlines()),
                    "bytes": path.stat().st_size,
                }
            )
            text = _without_comments(content)
            for match in INCLUDE_RE.finditer(text):
                target = _source_target(source, path, match.group(1))
                if target is None:
                    continue
                line = text.count("\n", 0, match.start()) + 1
                edges.append({"source": relative, "target": target})
                if target.startswith(METHOD_PREFIXES):
                    known = (relative, target) in KNOWN_METHOD_EDGES
                    method_edges.append(
                        {
                            "source": relative,
                            "target": target,
                            "line": line,
                            "known_debt": known,
                        }
                    )
                    if not known:
                        errors.append(
                            f"{relative}:{line}: forbidden {owner} dependency on {target}"
                        )

    duplicate = _duplicate_infrastructure(root)
    for name, item in duplicate.items():
        count = int(item["count"])
        allowed = int(item["allowed"])
        if count > allowed:
            errors.append(
                f"duplicate infrastructure debt grew for {name}: {count} > {allowed}"
            )

    return {
        "errors": errors,
        "modules": modules,
        "edges": edges,
        "method_edges": method_edges,
        "areas": _area_metrics(root),
        "duplicate_infrastructure": duplicate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit_electronic_structure_boundaries()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for error in report["errors"]:
            print(error, file=sys.stderr)
        debt = report["duplicate_infrastructure"]
        debt_text = ", ".join(
            f"{name}={item['count']}/{item['allowed']}" for name, item in debt.items()
        )
        known_edges = sum(bool(edge["known_debt"]) for edge in report["method_edges"])
        print(
            f"Checked {len(report['modules'])} shared native modules; "
            f"{len(report['edges'])} local dependency edges; "
            f"{len(report['errors'])} architecture errors; "
            f"{known_edges}/{len(KNOWN_METHOD_EDGES)} known reverse edges present; "
            f"known duplicate debt: {debt_text}"
        )
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
