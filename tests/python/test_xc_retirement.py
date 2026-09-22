"""Regression tests for the #745 handwritten-XC retirement boundary."""

import ast
from pathlib import Path

from tools.vibeqc_validation.xc_retirement import (
    LEGACY_MODULE_FILES,
    errors,
    inventory,
    scan_legacy_consumers,
    unexpected_consumers,
    unexpected_expression_modules,
)

ROOT = Path(__file__).resolve().parents[2]


def test_xc_retirement_gate_rejects_no_new_legacy_edges() -> None:
    assert unexpected_consumers(ROOT) == []
    assert unexpected_expression_modules(ROOT) == []
    assert errors(ROOT) == []


def test_xc_retirement_inventory_is_explicit_and_monotone() -> None:
    report = inventory(ROOT)
    assert report["schema"] == "vibeqc.xc-retirement-inventory.v1"
    consumers = scan_legacy_consumers(ROOT)
    assert len(report["consumers"]) == len(consumers)
    assert {row["module"] for row in report["legacy_sources"]} <= set(
        LEGACY_MODULE_FILES
    )


def test_retired_xc_handwritten_builders_stay_deleted() -> None:
    sources = (
        ROOT / "python/vibeqc_compiler/xc/expressions.py",
        ROOT / "python/vibeqc_compiler/xc/rsh_expressions.py",
    )
    function_names = set()
    for source in sources:
        tree = ast.parse(source.read_text(), filename=str(source.relative_to(ROOT)))
        function_names.update(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    retired = {
        "production_channel",
        "r2_switch",
        "scan_switch",
        "scan_exchange",
        "scan_correlation",
        "r2scan_exchange",
        "r2scan_correlation",
        "pw91_exchange",
        "pw91_correlation",
        "pw",
        "lda_correlation",
    }
    assert function_names.isdisjoint(retired)


def test_xc_retirement_gate_detects_new_consumer(tmp_path: Path) -> None:
    source = tmp_path / "python/vibeqc_compiler/xc/new_backend.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "from vibeqc_compiler.xc.wb97mv_expressions import energy_expression\n"
    )
    failures = errors(tmp_path)
    assert len(failures) == 1
    assert "new legacy XC consumer" in failures[0]


def test_xc_retirement_gate_detects_new_expression_module(tmp_path: Path) -> None:
    source = tmp_path / "python/vibeqc_compiler/xc/new_expressions.py"
    source.parent.mkdir(parents=True)
    source.write_text("def energy_expression():\n    return None\n")
    failures = errors(tmp_path)
    expected = (
        "python/vibeqc_compiler/xc/new_expressions.py: "
        "untracked handwritten-looking XC expression module"
    )
    assert failures == [expected]


def test_xc_retirement_final_gate_detects_remaining_consumer(tmp_path: Path) -> None:
    source = tmp_path / "python/vibeqc_compiler/xc/expression_dispatch.py"
    source.parent.mkdir(parents=True)
    source.write_text("from .expressions import energy_expression\n")
    failures = errors(tmp_path, require_no_consumers=True)
    assert failures == [
        (
            "python/vibeqc_compiler/xc/expression_dispatch.py:1: legacy XC consumer remains "
            "vibeqc_compiler.xc.expressions"
        )
    ]


def test_retired_direct_program_import_is_not_reauthorized(tmp_path: Path) -> None:
    """Moving the retained bridge must not allow the former consumer back in."""
    source = tmp_path / "python/vibeqc_compiler/xc/program.py"
    source.parent.mkdir(parents=True)
    source.write_text("from .expressions import energy_expression\n")
    newly_forbidden = (
        "python/vibeqc_compiler/xc/program.py:1: new legacy XC consumer "
        "vibeqc_compiler.xc.expressions (import-from)"
    )
    remaining = (
        "python/vibeqc_compiler/xc/program.py:1: legacy XC consumer remains "
        "vibeqc_compiler.xc.expressions"
    )
    assert errors(tmp_path) == [newly_forbidden]
    assert errors(tmp_path, require_no_consumers=True) == [newly_forbidden, remaining]
