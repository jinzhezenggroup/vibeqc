"""Enforce cross-method native ownership boundaries and report architecture metrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {".cpp", ".hpp", ".h", ".cu", ".cuh"}

# These layers are intended to remain reusable by every electronic-structure
# method. They may depend on one another and on chemistry/integral primitives,
# but never acquire new concrete HF/DFT/post-HF/CC implementation dependencies.
SHARED_OWNERS = ("core", "runtime", "tensor", "solver", "response")
# Some neutral contracts live in a mixed implementation directory. Protect the
# contract itself rather than incorrectly treating the whole directory as a
# shared layer.
SHARED_CONTRACTS = {
    "integrals/electron_interaction_source.hpp": "provider contract",
}
METHOD_PREFIXES = ("scf/", "dft/", "posthf/", "cc/")

# Current reverse edges are explicit debt ceilings, not approved design. The
# check allows them to disappear but rejects any new shared -> method edge.
KNOWN_METHOD_EDGES = {
    ("runtime/cuda_runtime.cu", "scf/aot_shell_registry.hpp"),
    ("runtime/host_component_trace.hpp", "scf/reference/observation.hpp"),
}

# Post-HF still reaches into these SCF-owned adapters and implementation
# headers. Freeze the exact current edge set so it may shrink while every new
# post-HF -> SCF dependency is rejected.
KNOWN_POSTHF_SCF_EDGES = {
    ("posthf/bridge.cpp", "scf/cuda_df_gradient.hpp"),
    ("posthf/bridge.cpp", "scf/cuda_one_electron_gradient.hpp"),
    ("posthf/bridge.cpp", "scf/cuda_weighted_eri.hpp"),
    ("posthf/bridge.cpp", "scf/density_fitting.hpp"),
    ("posthf/bridge.cpp", "scf/mean_field.hpp"),
    ("posthf/bridge.cpp", "scf/proposal_bridge.hpp"),
    ("posthf/bridge.cpp", "scf/proposals.hpp"),
    ("posthf/cuda_derivative.cpp", "scf/cuda_weighted_eri.hpp"),
    ("posthf/df_bridge.cu", "scf/cuda_density_fitting.hpp"),
    ("posthf/mp2_derivative_common.cpp", "scf/types.hpp"),
    ("posthf/mp2_derivative_cuda.cpp", "scf/cuda/rhf_policy.hpp"),
    ("posthf/mp2_derivative_cuda.cpp", "scf/cuda_one_electron_gradient.hpp"),
    ("posthf/mp2_energy.cpp", "scf/cuda_density_fitting_integrals.hpp"),
    ("posthf/mp2_energy.cpp", "scf/density_fitting.hpp"),
    ("posthf/mp2_force.cpp", "scf/types.hpp"),
    ("posthf/native_provider.hpp", "scf/types.hpp"),
    ("posthf/ri_mp2_cuda.cu", "scf/cuda/df_plan_internal.hpp"),
    ("posthf/ri_mp2_cuda.cu", "scf/cuda_density_fitting.hpp"),
    ("posthf/ri_mp2_cuda.cu", "scf/cuda_density_fitting_eigen.hpp"),
    ("posthf/ri_mp2_cuda.hpp", "scf/mean_field.hpp"),
}

# Host bounded iteration, DIIS, and its dense solve moved to src/solver in
# #920/#922. Their zero ceilings prevent the retired CC owners from returning.
# The method-specific CUDA DIIS loop remains explicit debt until it has a
# justified shared device contract.
DUPLICATE_POLICIES = {
    "cc_host_bounded_iteration_owner": {
        "kind": "function",
        "name": "run_bounded_iterations",
        "allowed_locations": (),
    },
    "cc_cpu_diis_owner": {
        "kind": "type",
        "name": "Diis",
        "allowed_locations": (),
    },
    "cc_cpu_local_linear_solver": {
        "kind": "function",
        "name": "solve_linear",
        "allowed_locations": (),
    },
    "cc_cuda_diis_owner": {
        "kind": "function",
        "name": "run_diis",
        "allowed_locations": ("cc/cuda_solver.cu",),
    },
}

# Canonical shared owners and the method areas that must keep consuming them.
# Consumers are derived from the repository include graph, including transitive
# forwarding headers. This makes a silent method-local fork visible even when a
# replacement uses different whitespace, comments or source layout.
SHARED_INFRASTRUCTURE = {
    "bounded_iteration": {
        "owner": "solver/iteration_control.hpp",
        "required_consumer_areas": ("cc", "scf"),
    },
    "host_diis": {
        "owner": "solver/diis.hpp",
        "required_consumer_areas": ("cc",),
    },
    "diis_coefficients": {
        "owner": "solver/diis_coefficients.hpp",
        "required_consumer_areas": ("cc", "scf"),
    },
    "diis_history": {
        "owner": "solver/diis_history.hpp",
        "required_consumer_areas": ("cc", "scf"),
    },
    "dense_linear": {
        "owner": "solver/dense_linear.hpp",
        "required_consumer_areas": ("cc", "scf"),
    },
    "electronic_reference": {
        "owner": "core/electronic_reference.hpp",
        "required_consumer_areas": ("dft", "scf"),
    },
    "electron_interaction_source": {
        "owner": "integrals/electron_interaction_source.hpp",
        "required_consumer_areas": ("posthf", "scf"),
    },
    "linear_response_problem": {
        "owner": "response/linear_problem.hpp",
        "required_consumer_areas": ("posthf",),
    },
}

# Stable pointers to the regression suites that carry the cross-method story.
# CI executes these suites separately; this checker keeps the owner/consumer
# inventory locatable and prevents a gate from disappearing without an explicit
# architecture update. It never inspects assertion text or sets LOC targets.
REGRESSION_GATES = {
    "hf_energy": {
        "consumers": ("hf",),
        "tests": (
            "tests/native/test_rhf.cpp",
            "tests/native/test_uhf.cpp",
        ),
    },
    "dft_energy": {
        "consumers": ("dft",),
        "tests": (
            "tests/native/test_dft.cpp",
            "tests/native/test_uks.cpp",
        ),
    },
    "cc_energy_amplitudes": {
        "consumers": ("cc",),
        "tests": (
            "tests/python/test_rccsd_public.py",
            "tests/python/test_cc_equations.py",
        ),
    },
    "forces": {
        "consumers": ("dft", "posthf", "cc"),
        "tests": (
            "tests/native/test_mp2_gradient.cpp",
            "tests/python/test_dft_stationary_gradient.py",
            "tests/python/test_cc_complete_gradient.py",
        ),
    },
    "response_hvp": {
        "consumers": ("hf", "dft", "posthf", "cc"),
        "tests": (
            "tests/native/test_native_gmres.cpp",
            "tests/python/test_hessian_hvp.py",
            "tests/python/test_response_uhf.py",
            "tests/python/test_response_native_rks.py",
            "tests/python/test_ccsd_t_orbital_response.py",
        ),
    },
    "memory_accounting": {
        "consumers": ("shared", "cc"),
        "tests": (
            "tests/native/test_runtime_workspace.cpp",
            "tests/python/test_rccsd_numeric_capacity.py",
        ),
    },
    "compiler_dependencies": {
        "consumers": ("method_ir", "execution_ir"),
        "tests": ("tests/python/test_compiler_structure.py",),
    },
    "runtime_dependencies": {
        "consumers": ("shared", "hf", "dft", "posthf", "cc"),
        "tests": ("tests/python/test_electronic_structure_boundaries.py",),
    },
    "shared_iteration_semantics": {
        "consumers": ("shared",),
        "tests": ("tests/native/test_self_consistent.cpp",),
    },
}

# Consume complete literals before recognizing comment delimiters. Number
# tokens protect C++ digit separators from being mistaken for character quotes.
CPP_NONCODE_RE = re.compile(
    r'(?P<raw>(?:u8|u|U|L)?R"(?P<delimiter>[^\s()\\]{0,16})\(.*?\)(?P=delimiter)")'
    r"|(?P<number>\b[0-9][\w.']*)"
    r'|(?P<quoted>"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')'
    r"|(?P<comment>/\*.*?\*/|//(?:\\\r?\n|[^\n])*)",
    re.DOTALL,
)
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]', re.MULTILINE)
CPP_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|::|->|[{}();<>\[\],&*:=]")


def _mask_non_newlines(text: str) -> str:
    return re.sub(r"[^\n]", " ", text)


def _without_comments(text: str) -> str:
    """Remove only real C++ comments, preserving literal bytes and line numbers."""
    return CPP_NONCODE_RE.sub(
        lambda match: (
            _mask_non_newlines(match.group())
            if match.group("comment") is not None
            else match.group()
        ),
        text,
    )


def _cpp_code(text: str, *, keep_include_paths: bool = False) -> str:
    """Mask literals as well, except quoted operands of actual include directives."""
    clean = _without_comments(text)

    def replacement(match: re.Match[str]) -> str:
        if match.group("number") is not None:
            return match.group()
        if (
            keep_include_paths
            and match.group("quoted") is not None
            and match.group().startswith('"')
        ):
            begin = clean.rfind("\n", 0, match.start()) + 1
            if re.fullmatch(
                r"[ \t]*#[ \t]*include[ \t]*", clean[begin : match.start()]
            ):
                return match.group()
        return _mask_non_newlines(match.group())

    return CPP_NONCODE_RE.sub(replacement, clean)


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
        (*(f"{owner}/" for owner in SHARED_OWNERS), *METHOD_PREFIXES)
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


def _metrics_by_group(
    root: Path, group_for: typing.Callable[[Path], str]
) -> dict[str, dict[str, int]]:
    source = root / "src"
    groups: dict[str, dict[str, int]] = {}
    for path in _native_files(root):
        relative = path.relative_to(source)
        group = group_for(relative)
        text = path.read_text(encoding="utf-8")
        metrics = groups.setdefault(
            group,
            {"files": 0, "lines": 0, "bytes": 0, "generated_lines": 0},
        )
        metrics["files"] += 1
        metrics["lines"] += len(text.splitlines())
        metrics["bytes"] += path.stat().st_size
        if "generated" in path.name:
            metrics["generated_lines"] += len(text.splitlines())
    return dict(sorted(groups.items()))


def _area_metrics(root: Path) -> dict[str, dict[str, int]]:
    """Return maintainability inventory without imposing size thresholds."""
    return _metrics_by_group(root, lambda relative: relative.parts[0])


def _ownership_metrics(root: Path) -> dict[str, dict[str, int]]:
    """Roll up shared/HF/DFT/CC areas as diagnostic metrics only."""

    def owner(relative: Path) -> str:
        spelling = relative.as_posix()
        area = relative.parts[0]
        if area in SHARED_OWNERS or spelling in SHARED_CONTRACTS:
            return "shared"
        if area == "scf":
            return "hf_only"
        if area == "dft":
            return "dft_only"
        if area == "cc":
            return "cc_only"
        if area == "posthf":
            return "posthf_only"
        return "other"

    return _metrics_by_group(root, owner)


def _cpp_tokens(text: str) -> list[tuple[str, int]]:
    """Lex identifiers/punctuation while excluding comments and literals."""
    clean = _cpp_code(text)
    tokens: list[tuple[str, int]] = []
    for match in CPP_TOKEN_RE.finditer(clean):
        token = match.group(0)
        if token.startswith(('"', "'")):
            continue
        tokens.append((token, clean.count("\n", 0, match.start()) + 1))
    return tokens


def _definition_locations(
    tokens: list[tuple[str, int]], kind: str, name: str
) -> list[int]:
    """Return lexical C++ definition lines, excluding calls and declarations."""
    locations: list[int] = []
    if kind == "type":
        for index, (token, _) in enumerate(tokens):
            if token not in {"class", "struct"}:
                continue
            if index and tokens[index - 1][0] in {"<", ","}:
                continue
            if index + 1 >= len(tokens) or tokens[index + 1][0] != name:
                continue
            name_line = tokens[index + 1][1]
            for candidate, _ in tokens[index + 2 :]:
                if candidate == ";":
                    break
                if candidate == "{":
                    locations.append(name_line)
                    break
        return locations

    for index, (token, line) in enumerate(tokens):
        if token != name or index + 1 >= len(tokens) or tokens[index + 1][0] != "(":
            continue
        if index and tokens[index - 1][0] in {
            "=",
            "return",
            "?",
            ":",
            "(",
            ",",
            "{",
            ";",
        }:
            continue
        depth = 0
        closing = None
        for cursor in range(index + 1, len(tokens)):
            candidate = tokens[cursor][0]
            if candidate == "(":
                depth += 1
            elif candidate == ")":
                depth -= 1
                if depth == 0:
                    closing = cursor
                    break
        if closing is None:
            continue
        for candidate, _ in tokens[closing + 1 :]:
            if candidate in {";", ")", ","}:
                break
            if candidate == "{":
                locations.append(line)
                break
    return locations


def _duplicate_infrastructure(root: Path) -> dict[str, dict[str, object]]:
    source = root / "src"
    cc = source / "cc"
    report: dict[str, dict[str, object]] = {}
    for name, policy in DUPLICATE_POLICIES.items():
        matches: list[str] = []
        if cc.is_dir():
            for path in sorted(cc.rglob("*")):
                if not path.is_file() or path.suffix not in SUFFIXES:
                    continue
                text = path.read_text(encoding="utf-8")
                tokens = _cpp_tokens(text)
                lines = set(
                    _definition_locations(
                        tokens, str(policy["kind"]), str(policy["name"])
                    )
                )
                for line in sorted(lines):
                    matches.append(f"{path.relative_to(source).as_posix()}:{line}")
        allowed_locations = list(policy["allowed_locations"])
        report[name] = {
            "count": len(matches),
            "allowed": len(allowed_locations),
            "allowed_locations": allowed_locations,
            "locations": matches,
        }
    return report


def _native_dependency_edges(root: Path) -> list[dict[str, object]]:
    source = (root / "src").resolve()
    edges: list[dict[str, object]] = []
    for path in _native_files(root):
        relative = path.relative_to(source).as_posix()
        text = _cpp_code(path.read_text(encoding="utf-8"), keep_include_paths=True)
        for match in INCLUDE_RE.finditer(text):
            target = _source_target(source, path, match.group(1))
            if target is None:
                continue
            edges.append(
                {
                    "source": relative,
                    "target": target,
                    "line": text.count("\n", 0, match.start()) + 1,
                }
            )
    return edges


def _protected_owner(relative: str) -> str | None:
    area = relative.split("/", 1)[0]
    if area in SHARED_OWNERS:
        return area
    return SHARED_CONTRACTS.get(relative)


def _infrastructure_inventory(
    root: Path, dependency_edges: list[dict[str, object]], errors: list[str]
) -> dict[str, dict[str, object]]:
    source = root / "src"
    reverse: dict[str, set[str]] = {}
    for edge in dependency_edges:
        reverse.setdefault(str(edge["target"]), set()).add(str(edge["source"]))
    full_checkout = (root / "pyproject.toml").is_file()
    inventory: dict[str, dict[str, object]] = {}
    for name, specification in SHARED_INFRASTRUCTURE.items():
        owner = str(specification["owner"])
        required = list(specification["required_consumer_areas"])
        direct_consumers = set(reverse.get(owner, set()))
        consumers: set[str] = set()
        pending = list(direct_consumers)
        while pending:
            consumer = pending.pop()
            if consumer in consumers:
                continue
            consumers.add(consumer)
            pending.extend(reverse.get(consumer, set()) - consumers)
        consumer_areas = sorted({path.split("/", 1)[0] for path in consumers})
        owner_exists = (source / owner).is_file()
        inventory[name] = {
            "owner": owner,
            "owner_exists": owner_exists,
            "required_consumer_areas": required,
            "consumer_areas": consumer_areas,
            "direct_consumers": sorted(direct_consumers),
            "transitive_consumers": sorted(consumers - direct_consumers),
            "consumers": sorted(consumers),
        }
        if not full_checkout:
            continue
        if not owner_exists:
            errors.append(f"shared infrastructure owner missing for {name}: {owner}")
        for area in required:
            if area not in consumer_areas:
                errors.append(
                    f"shared infrastructure {name} ({owner}) lost required {area} consumer"
                )
    return inventory


def _regression_inventory(
    root: Path, errors: list[str]
) -> dict[str, dict[str, object]]:
    full_checkout = (root / "pyproject.toml").is_file()
    inventory: dict[str, dict[str, object]] = {}
    for name, specification in REGRESSION_GATES.items():
        tests = [
            {"path": path, "exists": (root / path).is_file()}
            for path in specification["tests"]
        ]
        inventory[name] = {
            "consumers": list(specification["consumers"]),
            "tests": tests,
        }
        if full_checkout:
            for test in tests:
                if not test["exists"]:
                    errors.append(
                        f"regression inventory missing {name} gate: {test['path']}"
                    )
    return inventory


def audit_electronic_structure_boundaries(root: Path = ROOT) -> dict[str, object]:
    """Audit reusable native layers against concrete method dependencies."""
    source = (root / "src").resolve()
    errors: list[str] = []
    edges: list[dict[str, object]] = []
    method_edges: list[dict[str, object]] = []
    posthf_scf_edges: list[dict[str, object]] = []
    modules: list[dict[str, object]] = []

    dependency_edges = _native_dependency_edges(root)
    edges_by_source: dict[str, list[dict[str, object]]] = {}
    for edge in dependency_edges:
        edges_by_source.setdefault(str(edge["source"]), []).append(edge)

    for path in _native_files(root):
        relative = path.relative_to(source).as_posix()
        owner = _protected_owner(relative)
        if owner is None:
            continue
        content = path.read_text(encoding="utf-8")
        modules.append(
            {
                "owner": owner,
                "path": relative,
                "lines": len(content.splitlines()),
                "bytes": path.stat().st_size,
            }
        )
        for edge in edges_by_source.get(relative, []):
            target = str(edge["target"])
            line = int(edge["line"])
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

    for edge in dependency_edges:
        relative = str(edge["source"])
        target = str(edge["target"])
        if not relative.startswith("posthf/") or not target.startswith("scf/"):
            continue
        line = int(edge["line"])
        known = (relative, target) in KNOWN_POSTHF_SCF_EDGES
        posthf_scf_edges.append(
            {
                "source": relative,
                "target": target,
                "line": line,
                "known_debt": known,
            }
        )
        if not known:
            errors.append(
                f"{relative}:{line}: forbidden post-HF dependency on SCF-owned {target}"
            )

    duplicate = _duplicate_infrastructure(root)
    for name, item in duplicate.items():
        count = int(item["count"])
        allowed = int(item["allowed"])
        if count > allowed:
            locations = ", ".join(str(location) for location in item["locations"])
            errors.append(
                f"duplicate infrastructure debt grew for {name}: {count} > {allowed} "
                f"at {locations}"
            )
        allowed_locations = {str(path) for path in item["allowed_locations"]}
        unapproved = [
            str(location)
            for location in item["locations"]
            if str(location).rsplit(":", 1)[0] not in allowed_locations
        ]
        if allowed_locations and unapproved:
            errors.append(f"{name} has an unapproved owner at {', '.join(unapproved)}")

    infrastructure = _infrastructure_inventory(root, dependency_edges, errors)
    regression = _regression_inventory(root, errors)

    return {
        "errors": errors,
        "modules": modules,
        "edges": edges,
        "method_edges": method_edges,
        "posthf_scf_edges": posthf_scf_edges,
        "areas": _area_metrics(root),
        "ownership_groups": _ownership_metrics(root),
        "duplicate_infrastructure": duplicate,
        "infrastructure_inventory": infrastructure,
        "regression_inventory": regression,
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
        known_posthf_edges = sum(
            bool(edge["known_debt"]) for edge in report["posthf_scf_edges"]
        )
        print(
            f"Checked {len(report['modules'])} shared native modules; "
            f"{len(report['edges'])} local dependency edges; "
            f"{len(report['errors'])} architecture errors; "
            f"{known_edges}/{len(KNOWN_METHOD_EDGES)} known reverse edges present; "
            f"{known_posthf_edges}/{len(KNOWN_POSTHF_SCF_EDGES)} known post-HF -> SCF edges present; "
            f"known duplicate debt: {debt_text}"
        )
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
