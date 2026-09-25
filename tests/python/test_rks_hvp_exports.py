"""Exercise the actual lazy export table without loading numerical dependencies."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("order", ["function", "result", "implementation"])
@pytest.mark.parametrize("repeat", [1, 2])
def test_rks_hvp_export_is_independent_of_import_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: str, repeat: int
) -> None:
    # Copy only the real export machinery into an isolated package. The stub
    # implementation exercises Python import binding, not Hessian mathematics.
    source = (ROOT / "tools/vibeqc_hessian/__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    lazy = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_LAZY"
            for target in node.targets
        )
    )
    getter = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
    )
    mapping = ast.literal_eval(lazy.value)
    name = f"hvp_export_{order}_{repeat}"
    package = tmp_path / name
    package.mkdir()
    (package / "__init__.py").write_text(
        "import typing\n" + ast.unparse(lazy) + "\n" + ast.unparse(getter) + "\n",
        encoding="utf-8",
    )
    implementation = mapping["rks_hvp"]
    (package / f"{implementation}.py").write_text(
        "class RKSHVPResult: pass\ndef rks_hvp(): return RKSHVPResult()\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        module = importlib.import_module(name)
        if order == "result":
            getattr(module, "RKSHVPResult")
        elif order == "implementation":
            importlib.import_module(f"{name}.{implementation}")
        for _ in range(repeat):
            exported = __import__(name, fromlist=["rks_hvp", "RKSHVPResult"])
            assert callable(exported.rks_hvp)
            assert isinstance(exported.rks_hvp(), exported.RKSHVPResult)
    finally:
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]
