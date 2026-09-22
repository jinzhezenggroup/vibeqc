"""The pooled-J/K benchmark must reject invalid results before publication."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def run_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    energies: list[float],
    converged: bool = True,
) -> None:
    good = SimpleNamespace(
        energy=-75.0, converged=True, iterations=1, executed_backend="cuda"
    )
    items = [
        SimpleNamespace(
            energy=e, converged=converged, iterations=1, executed_backend="cuda"
        )
        for e in energies
    ]

    class Batch:
        calls = 0

        def __enter__(self) -> "Batch":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def set_warm_start_updates(self, value: bool) -> None:
            assert value is False

        def execute(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs == {"strict": True, "properties": ("energy",)}
            self.calls += 1
            return SimpleNamespace(items=[good, good] if self.calls < 3 else items)

    class Calculator:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["device"] == "cuda"

        def prepare_batch(self, systems: list[object], **kwargs: object) -> Batch:
            assert len(systems) == 2 and kwargs == {"warm_start": True}
            return Batch()

    def fake_module(name: str, **attributes: object) -> None:
        module = ModuleType(name)
        for key, value in attributes.items():
            monkeypatch.setattr(module, key, value, raising=False)
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setenv("SLURM_JOB_ID", "isolated-test-no-GPU")
    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--batch-sizes", "2", "--repeats", "1"]
    )
    fake_module("vibeqc", Calculator=Calculator)
    fake_module(
        "cupy",
        cuda=SimpleNamespace(
            Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None))
        ),
    )
    fake_module(
        "_support",
        environment_metadata=lambda **kwargs: {},
        raw_output_path=Path,
        write_result=lambda *args: pytest.fail("unexpected file write"),
    )
    spec = importlib.util.spec_from_file_location(
        "pooled_jk_probe", ROOT / "benchmarks/issue992_pooled_direct_jk.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_energy_cannot_pass_spread_gate(
    monkeypatch: pytest.MonkeyPatch, bad: float
) -> None:
    with pytest.raises(RuntimeError, match="finite"):
        run_benchmark(monkeypatch, [bad, bad])


@pytest.mark.parametrize("count", [1, 3])
def test_wrong_batch_count_is_not_a_throughput_result(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    with pytest.raises(RuntimeError, match="count"):
        run_benchmark(monkeypatch, [-75.0] * count)


def test_unconverged_replay_is_not_published(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="converge"):
        run_benchmark(monkeypatch, [-75.0, -75.0], converged=False)


def test_valid_replay_keeps_original_result_contract(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_benchmark(monkeypatch, [-75.0, -75.0])
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"]["2"]["iterations"] == [[1, 1]]
    assert payload["settings"]["properties"] == ["energy"]
