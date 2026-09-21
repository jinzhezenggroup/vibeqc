"""Exercise the real endpoint wrapper without requiring a CUDA allocation."""

import ast
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import df_policy_endpoint as endpoint


def _execute() -> typing.Any:
    source = Path(endpoint.__file__).read_text()
    tree = ast.parse(source)
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "execute"
    )
    namespace = {**vars(endpoint), "args": SimpleNamespace(energy_only=False)}
    # Execute only the AST of the checked-in function under test, never external input.
    exec(  # noqa: S102
        compile(
            ast.Module(body=[node], type_ignores=[]), str(endpoint.__file__), "exec"
        ),
        namespace,
    )
    return namespace["execute"]


@pytest.mark.parametrize("inherited", [None, "1", "off"])
def test_clean_and_cold_calls_disable_inherited_screening(
    monkeypatch: pytest.MonkeyPatch, inherited: str | None
) -> None:
    if inherited is None:
        monkeypatch.delenv("VIBEQC_DF_SCREENING_FEATURES", raising=False)
    else:
        monkeypatch.setenv("VIBEQC_DF_SCREENING_FEATURES", inherited)
    seen = []
    owner = SimpleNamespace(
        execute=lambda **kw: seen.append(
            endpoint.os.environ.get("VIBEQC_DF_SCREENING_FEATURES")
        )
    )
    execute = _execute()
    execute(owner)
    execute(owner, cold=True)
    assert seen == [None, None]
    assert endpoint.os.environ.get("VIBEQC_DF_SCREENING_FEATURES") == inherited


@pytest.mark.parametrize("inherited", [None, "off"])
def test_diagnostic_failure_restores_original_environment(
    monkeypatch: pytest.MonkeyPatch, inherited: str | None
) -> None:
    if inherited is None:
        monkeypatch.delenv("VIBEQC_DF_SCREENING_FEATURES", raising=False)
    else:
        monkeypatch.setenv("VIBEQC_DF_SCREENING_FEATURES", inherited)

    def fail(**kw: typing.Any) -> typing.NoReturn:
        assert endpoint.os.environ.get("VIBEQC_DF_SCREENING_FEATURES") == "1"
        raise RuntimeError("injected endpoint failure")

    with pytest.raises(RuntimeError, match="injected endpoint failure"):
        _execute()(SimpleNamespace(execute=fail), screening_features=True)
    assert endpoint.os.environ.get("VIBEQC_DF_SCREENING_FEATURES") == inherited
