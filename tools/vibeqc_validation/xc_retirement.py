"""Source-level inventory and regression gate for XC formula retirement (#745).

The gate is intentionally syntax-only. It does not import VibeQC, Libxc, CUDA, or
native code, so it can run in ordinary CPU CI while family-specific numerical
qualification proceeds independently.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path

LEGACY_MODULE_FILES = {
    "vibeqc_compiler.xc.expressions": "python/vibeqc_compiler/xc/expressions.py",
    "vibeqc_compiler.xc.rsh_expressions": (
        "python/vibeqc_compiler/xc/rsh_expressions.py"
    ),
    "vibeqc_compiler.xc.wb97mv_expressions": (
        "python/vibeqc_compiler/xc/wb97mv_expressions.py"
    ),
}

# This is a ceiling, not a list that must stay populated. Removing any edge is
# always allowed. Adding a new edge fails CI and therefore cannot silently turn
# a qualification oracle back into a production compatibility backend.
LEGACY_CONSUMER_CEILING = {
    (
        "python/vibeqc_compiler/xc/expression_dispatch.py",
        "vibeqc_compiler.xc.rsh_expressions",
    ),
    ("tests/python/test_libxc_maple_pw91.py", "vibeqc_compiler.xc.rsh_expressions"),
    ("tests/python/test_libxc_maple_lyp.py", "vibeqc_compiler.xc.rsh_expressions"),
    # Qualification oracles already present in the integration base (6b965bd7).
    # These do not admit any additional production/runtime consumer.
    ("tests/python/test_libxc_maple_b88.py", "vibeqc_compiler.xc.rsh_expressions"),
    ("tests/python/test_libxc_maple_p86_pz.py", "vibeqc_compiler.xc.rsh_expressions"),
    ("tests/python/test_libxc_maple_vwn.py", "vibeqc_compiler.xc.rsh_expressions"),
}

SCAN_ROOTS = ("python", "tools", "tests")


@dataclass(frozen=True, order=True)
class LegacyConsumer:
    """One source-level dependency on a handwritten XC expression module."""

    path: str
    module: str
    line: int
    kind: str

    @property
    def key(self) -> tuple[str, str]:
        return self.path, self.module


def _relative_module(path: Path, node: ast.ImportFrom) -> str:
    module = node.module or ""
    if node.level == 0:
        return module
    parts = path.with_suffix("").parts
    if len(parts) < 3 or parts[0] != "python" or parts[1] != "vibeqc_compiler":
        return module
    package = ".".join(parts[1:-1])
    return importlib.util.resolve_name("." * node.level + module, package)


def _dynamic_module(node: ast.Call) -> str | None:
    function = node.func
    named = isinstance(function, ast.Name) and function.id == "import_module"
    qualified = (
        isinstance(function, ast.Attribute)
        and function.attr == "import_module"
        and isinstance(function.value, ast.Name)
        and function.value.id == "importlib"
    )
    builtin = isinstance(function, ast.Name) and function.id == "__import__"
    if not (named or qualified or builtin):
        return None
    name_node = (
        node.args[0]
        if node.args
        else next((item.value for item in node.keywords if item.arg == "name"), None)
    )
    if not isinstance(name_node, ast.Constant) or not isinstance(name_node.value, str):
        return None
    value = name_node.value
    if builtin:
        # Only literal absolute built-in imports are resolved here. Relative
        # __import__ needs a globals/package environment, not import_module's
        # second-argument package convention.
        level = (
            node.args[4]
            if len(node.args) > 4
            else next(
                (item.value for item in node.keywords if item.arg == "level"), None
            )
        )
        if level is not None and (
            not isinstance(level, ast.Constant) or level.value != 0
        ):
            return None
        return value if not value.startswith(".") else None
    if value.startswith("."):
        package_node = (
            node.args[1]
            if len(node.args) > 1
            else next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "package"
                ),
                None,
            )
        )
        if not isinstance(package_node, ast.Constant) or not isinstance(
            package_node.value, str
        ):
            return None
        try:
            return importlib.util.resolve_name(value, package_node.value)
        except ImportError:
            # This invalid literal import cannot resolve a live legacy module.
            return None
    return value


def scan_legacy_consumers(root: Path) -> list[LegacyConsumer]:
    """Find current code/test dependencies on the three legacy XC modules."""

    consumers: set[LegacyConsumer] = set()
    for scan_root in SCAN_ROOTS:
        base = root / scan_root
        if not base.exists():
            continue
        for source in sorted(base.rglob("*.py")):
            relative = source.relative_to(root)
            tree = ast.parse(source.read_text(), filename=str(relative))
            for node in ast.walk(tree):
                modules: list[tuple[str, str]] = []
                if isinstance(node, ast.ImportFrom):
                    module = _relative_module(relative, node)
                    modules.append((module, "import-from"))
                    modules.extend(
                        (f"{module}.{alias.name}", "import-from")
                        for alias in node.names
                        if module and alias.name != "*"
                    )
                elif isinstance(node, ast.Import):
                    modules.extend((alias.name, "import") for alias in node.names)
                elif isinstance(node, ast.Call):
                    module = _dynamic_module(node)
                    if module is not None:
                        modules.append((module, "dynamic-import"))
                for module, kind in modules:
                    if module in LEGACY_MODULE_FILES:
                        consumers.add(
                            LegacyConsumer(
                                path=relative.as_posix(),
                                module=module,
                                line=node.lineno,
                                kind=kind,
                            )
                        )
    return sorted(consumers)


def unexpected_consumers(root: Path) -> list[LegacyConsumer]:
    """Return consumer edges added beyond the migration-time ceiling."""

    return [
        consumer
        for consumer in scan_legacy_consumers(root)
        if consumer.key not in LEGACY_CONSUMER_CEILING
    ]


def legacy_sources(root: Path) -> list[dict[str, object]]:
    """Report remaining legacy source files without interpreting their formulas."""

    rows = []
    for module, relative in LEGACY_MODULE_FILES.items():
        path = root / relative
        if not path.exists():
            continue
        text = path.read_text()
        rows.append(
            {
                "module": module,
                "path": relative,
                "bytes": path.stat().st_size,
                "lines": len(text.splitlines()),
            }
        )
    return rows


def unexpected_expression_modules(root: Path) -> list[str]:
    """Reject newly introduced handwritten-looking expression modules."""

    xc = root / "python/vibeqc_compiler/xc"
    if not xc.exists():
        return []
    allowed = set(LEGACY_MODULE_FILES.values())
    return sorted(
        path.relative_to(root).as_posix()
        for path in xc.rglob("*expressions.py")
        if path.relative_to(root).as_posix() not in allowed
    )


def inventory(root: Path) -> dict[str, object]:
    """Return deterministic retirement status suitable for CI or issue evidence."""

    consumers = scan_legacy_consumers(root)
    return {
        "schema": "vibeqc.xc-retirement-inventory.v1",
        "legacy_sources": legacy_sources(root),
        "consumers": [asdict(consumer) for consumer in consumers],
        "unexpected_consumers": [
            asdict(consumer) for consumer in unexpected_consumers(root)
        ],
        "unexpected_expression_modules": unexpected_expression_modules(root),
    }


def errors(root: Path, *, require_no_consumers: bool = False) -> list[str]:
    """Return structural violations without running numerical code."""

    result = []
    for consumer in unexpected_consumers(root):
        result.append(
            f"{consumer.path}:{consumer.line}: new legacy XC consumer "
            f"{consumer.module} ({consumer.kind})"
        )
    for path in unexpected_expression_modules(root):
        result.append(f"{path}: untracked handwritten-looking XC expression module")
    if require_no_consumers:
        for consumer in scan_legacy_consumers(root):
            result.append(
                f"{consumer.path}:{consumer.line}: legacy XC consumer remains "
                f"{consumer.module}"
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--require-no-consumers",
        action="store_true",
        help="fail until every production/test legacy-module dependency is retired",
    )
    parser.add_argument("--json", action="store_true", help="print inventory JSON")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.json:
        print(json.dumps(inventory(root), indent=2, sort_keys=True))
    failures = errors(root, require_no_consumers=args.require_no_consumers)
    for failure in failures:
        print(failure)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
