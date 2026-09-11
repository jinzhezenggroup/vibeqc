"""Lightweight source-level enforcement of compiler ownership directions."""

import ast
import importlib.util
from pathlib import Path

from .paths import PACKAGE

# Generic backend services never reach into a scientific subsystem. TensorIR
# and IntegralIR stay independent; XC reuses scalar algebra and DFT ingredients.
ALLOWED = {
    "common": {"common"},
    "integral": {"integral", "common"},
    "tensor": {"tensor", "common"},
    "dft": {"dft", "common"},
    "xc": {"xc", "integral", "dft", "common"},
}

# These existing adapters consume the public molecular/native ABI only when
# invoked. Neither importing them nor generating mathematical source loads it.
RUNTIME_ADAPTERS = {
    "dft.ao": {"Calculator", "Primitive", "Shell", "_native"},
    "dft.grid": {"Atom"},
    "dft.fixtures": {"Primitive", "Shell"},
}

# Scalar algebra/emission still has its original IntegralIR package location.
# AO lowering reuses exactly these neutral facilities, as XC already does;
# it must not acquire integral recurrence, SCF, or schedule dependencies.
SCALAR_CLIENTS = {
    "dft.ao_cuda": {
        "vibeqc_compiler.integral.expr",
        "vibeqc_compiler.integral.cuda",
    },
}


def audit_structure(package: Path = PACKAGE) -> dict:
    """Report forbidden import edges and module sizes without importing code.

    A nested import still counts as a dependency. The only user-runtime
    exceptions are the enumerated lazy native adapters above. Relative imports
    are resolved before checking, so spelling changes cannot evade the gate.
    """
    errors, edges, modules = [], set(), []
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package)
        parts = list(relative.with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        if not parts:
            continue
        owner = parts[0]
        if owner not in ALLOWED:
            errors.append(f"{relative}: unknown compiler owner {owner}")
            continue
        name = ".".join(parts)
        parent = "vibeqc_compiler." + ".".join(parts if is_package else parts[:-1])
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
        modules.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "lines": len(text.splitlines()),
            }
        )

        # Track whether each import lives inside an explicit operation. Class
        # bodies and conditionals at module scope still execute during import.
        def visit(
            node,
            lazy=False,
            *,
            parent=parent,
            relative=relative,
            owner=owner,
            name=name,
        ):
            lazy = lazy or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            if isinstance(node, ast.ImportFrom):
                target = node.module or ""
                if node.level:
                    target = importlib.util.resolve_name(
                        "." * node.level + target, parent
                    )
                targets = [target]
                if target == "vibeqc_compiler":
                    targets = [target + "." + alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            else:
                targets = []
            for target in targets:
                location = f"{relative}:{node.lineno}"
                if target.startswith("vibeqc_compiler."):
                    destination = target.split(".")[1]
                    edges.add((owner, destination))
                    if destination not in ALLOWED[
                        owner
                    ] and target not in SCALAR_CLIENTS.get(name, set()):
                        errors.append(
                            f"{location}: forbidden {owner} -> {destination} import"
                        )
                elif target.split(".")[0] in {
                    "tools",
                    "benchmarks",
                    "pyscf",
                    "torch",
                    "cupy",
                }:
                    errors.append(
                        f"{location}: compiler imports script/reference dependency {target}"
                    )
                elif target.split(".")[0] == "vibeqc":
                    allowed = RUNTIME_ADAPTERS.get(name, set())
                    if not (
                        lazy
                        and isinstance(node, ast.ImportFrom)
                        and target == "vibeqc"
                        and {a.name for a in node.names} <= allowed
                    ):
                        errors.append(
                            f"{location}: compiler imports user runtime outside a lazy native adapter"
                        )
            for child in ast.iter_child_nodes(node):
                visit(child, lazy)

        visit(tree)
    return {"errors": errors, "edges": sorted(edges), "modules": modules}
