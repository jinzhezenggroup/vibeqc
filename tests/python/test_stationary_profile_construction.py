"""Execute the actual constructor expressions before any topology upload."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("call_index", (0, 1))
@pytest.mark.parametrize("profile_device", (False, True))
def test_cold_constructor_receives_requested_profile(
    call_index: int, profile_device: bool
) -> None:
    tree = ast.parse((ROOT / "python/vibeqc/_stationary_cuda.py").read_text())
    calls = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_CudaSources"
        ),
        key=lambda node: node.lineno,
    )
    assert len(calls) == 2
    call = calls[call_index]
    names = {node.id for node in ast.walk(call) if isinstance(node, ast.Name)}
    context = {name: None for name in names}
    context.update(plan=SimpleNamespace(spin_blocks=1), profile_device=profile_device)

    def construct(
        *args: object, profile_device: bool = False, **kwargs: object
    ) -> bool:
        # The real constructor performs topology work before it returns.
        return profile_device

    def argument(node: ast.expr) -> object:
        if isinstance(node, ast.Name):
            return context[node.id]
        if isinstance(node, ast.Attribute):
            return getattr(argument(node.value), node.attr)
        if isinstance(node, ast.Constant):
            return node.value
        raise AssertionError("unexpected constructor argument expression")

    positional = [argument(node) for node in call.args]
    keywords = {item.arg: argument(item.value) for item in call.keywords}
    result = construct(*positional, **keywords)
    assert result is profile_device
