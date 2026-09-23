"""The packed/factorized comparison must explicitly request the same producer."""

from __future__ import annotations

import ast
import os
import typing
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def probe() -> dict[str, typing.Any]:
    # Execute the probe's actual control owner, without importing its GPU/oracle
    # dependencies or replacing its implementation with a test-side copy.
    path = ROOT / "benchmarks/df_admission_probe.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"CONTROLS", "VARIANTS", "TRACE_CONTROLS"}
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in names
                for target in node.targets
            )
        )
        or isinstance(node, ast.FunctionDef)
        and node.name == "controls"
    ]
    assert len(selected) == 4
    namespace = {"os": os, "typing": typing, "contextmanager": contextmanager}
    code = compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec")
    # Only selected AST nodes from the trusted repository fixture run here.
    exec(code, namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize("variant", ("auto", "shell", "packet", "packed", "factorized"))
@pytest.mark.parametrize("fail", (False, True))
def test_probe_controls_isolate_and_restore_response_space(
    probe: dict[str, typing.Any],
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    fail: bool,
) -> None:
    key = "VIBEQC_DF_RESPONSE_SPACE"
    monkeypatch.setenv(key, "dense")
    monkeypatch.setenv("VIBEQC_DF_RESPONSE_FUSION", "caller-policy")
    before = dict(os.environ)
    try:
        with probe["controls"](variant):
            expected = "occupied" if variant in ("packed", "factorized") else None
            actual = os.environ.get(key)
            assert actual == expected
            if fail:
                raise RuntimeError("injected endpoint failure")
    except RuntimeError as error:
        assert fail and str(error) == "injected endpoint failure"
    assert dict(os.environ) == before


def test_factorized_probe_changes_only_fusion_relative_to_packed(
    probe: dict[str, typing.Any],
) -> None:
    controls, variants = probe["CONTROLS"], probe["VARIANTS"]
    packed = dict(zip(controls, variants["packed"], strict=True))
    factorized = dict(zip(controls, variants["factorized"], strict=True))
    assert packed["RESPONSE_SPACE"] == factorized["RESPONSE_SPACE"] == "occupied"
    assert {key for key in packed if packed[key] != factorized[key]} == {
        "RESPONSE_FUSION"
    }
