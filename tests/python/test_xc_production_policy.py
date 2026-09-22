"""Structural gates for the #745 production-policy split."""

import ast
from pathlib import Path

from tools.vibeqc_validation.xc_retirement import scan_legacy_consumers

ROOT = Path(__file__).resolve().parents[2]


def _function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_production_policy_is_separate_from_legacy_formula_dispatch() -> None:
    legacy = ROOT / "python/vibeqc_compiler/xc/expressions.py"
    policy = ROOT / "python/vibeqc_compiler/xc/production_policy.py"
    policy_names = _function_names(policy)
    assert {
        "lda_xc_pw_unpolarized_tail_expression",
        "lda_xc_pw_polarized_tail_expression",
        "pbe_correlation_scaled_expression",
        "pbe_exchange_direct_expression",
        "pbe_exchange_reciprocal_expression",
    } <= policy_names
    legacy_text = legacy.read_text()
    assert "_PW_PARAMETERS" not in legacy_text
    assert {"pw", "lda_correlation"}.isdisjoint(_function_names(legacy))


def test_cpu_generator_has_no_legacy_expression_consumer_edge() -> None:
    assert all(
        consumer.path != "tools/generate_xc_cpu.py"
        for consumer in scan_legacy_consumers(ROOT)
    )
