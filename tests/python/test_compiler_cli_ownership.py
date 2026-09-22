from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPILER_ROOT = ROOT / "python" / "vibeqc_compiler"


def _tool_imports(root: Path) -> list[tuple[Path, int, str]]:
    violations: list[tuple[Path, int, str]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: tuple[str, ...]
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules = (node.module or "",) if node.level == 0 else ()
            else:
                continue
            for module in modules:
                if module == "tools" or module.startswith("tools."):
                    violations.append((path, node.lineno, module))
    return violations


def test_compiler_library_does_not_import_repository_tools() -> None:
    assert COMPILER_ROOT.is_dir()
    violations = _tool_imports(COMPILER_ROOT)
    detail = "\n".join(
        f"{path.relative_to(ROOT)}:{line}: imports {module}"
        for path, line, module in violations
    )
    assert not violations, (
        "vibeqc_compiler must own reusable compiler logic; repository tools/CLI "
        f"may consume it but must not become library dependencies:\n{detail}"
    )


def test_tool_import_detector_accepts_library_dependencies(tmp_path: Path) -> None:
    package = tmp_path / "vibeqc_compiler"
    package.mkdir()
    (package / "clean.py").write_text(
        "from .common import identity\nimport json\n", encoding="utf-8"
    )
    assert _tool_imports(package) == []


def test_tool_import_detector_rejects_cli_dependency(tmp_path: Path) -> None:
    package = tmp_path / "vibeqc_compiler"
    package.mkdir()
    bad = package / "bad.py"
    bad.write_text(
        "from tools.generate_xc_cpu import build_roots\nimport tools.source_registry\n",
        encoding="utf-8",
    )
    assert _tool_imports(package) == [
        (bad, 1, "tools.generate_xc_cpu"),
        (bad, 2, "tools.source_registry"),
    ]
