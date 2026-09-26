"""Production ownership gates for method-neutral response contracts."""

import ast
from pathlib import Path

import vibeqc.response.operators as production_operators
import vibeqc.response.problem as production_problem

from tools.vibeqc_response import operators as compatibility_operators
from tools.vibeqc_response import problem as compatibility_problem
from tools.vibeqc_response.native_ks import NativeRKSResponse


def _imported_modules(module: object) -> tuple[str, ...]:
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    return tuple(imported)


def test_production_response_contracts_do_not_import_tools() -> None:
    for module in (production_problem, production_operators):
        assert not tuple(
            name
            for name in _imported_modules(module)
            if name == "tools" or name.startswith("tools.")
        )


def test_tools_problem_shim_reexports_production_classes() -> None:
    assert (
        compatibility_problem.ResponseCompatibilityError
        is production_problem.ResponseCompatibilityError
    )
    assert compatibility_problem.ResponseProblem is production_problem.ResponseProblem
    assert (
        compatibility_problem.ResponseSolveError
        is production_problem.ResponseSolveError
    )
    assert (
        compatibility_problem.ResponseUnsupported
        is production_problem.ResponseUnsupported
    )
    assert compatibility_problem.RotationLayout is production_problem.RotationLayout


def test_tools_operator_shim_reexports_production_classes() -> None:
    assert (
        compatibility_operators.CPKSResponseOperator
        is production_operators.CPKSResponseOperator
    )
    assert (
        compatibility_operators.DenseMatrixResponseOperator
        is production_operators.DenseMatrixResponseOperator
    )
    assert (
        compatibility_operators.RHFResponseOperator
        is production_operators.RHFResponseOperator
    )
    assert (
        compatibility_operators.cpks_operator_identity
        is production_operators.cpks_operator_identity
    )
    assert (
        compatibility_operators.rhf_operator_identity
        is production_operators.rhf_operator_identity
    )


def test_native_rks_adapter_consumes_production_cpks_operator() -> None:
    assert issubclass(NativeRKSResponse, production_operators.CPKSResponseOperator)
