"""The pooled benchmark rejects invalid evidence before publishing throughput."""

import json
import runpy
import sys
import typing
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def run_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    *,
    phase: int = -1,
    defect: str = "",
    batch_size: int = 4,
) -> None:
    """Exercise the actual CLI with synthetic results, without a GPU or library."""
    call = 0

    def execute(**kwargs: typing.Any) -> SimpleNamespace:
        nonlocal call
        assert kwargs == {"strict": True, "properties": ("energy",)}
        count = batch_size
        if call == phase:
            count += {"missing": -1, "extra": 1}.get(defect, 0)
        items = [
            SimpleNamespace(
                converged=True, energy=-75.0, iterations=3, executed_backend="cuda"
            )
            for _ in range(count)
        ]
        if call == phase:
            if defect in ("nan", "inf", "-inf"):
                items[-1].energy = float(defect)
            elif defect == "unconverged":
                items[-1].converged = False
            elif defect in ("cpu_reference", "unknown_backend"):
                items[-1].executed_backend = defect
            elif defect == "missing_backend":
                del items[-1].executed_backend
            elif defect == "spread":
                items[-1].energy += 1e-6
        call += 1
        return SimpleNamespace(items=items)

    @contextmanager
    def prepare(
        systems: list[typing.Any], *, warm_start: bool
    ) -> typing.Iterator[SimpleNamespace]:
        assert len(systems) == batch_size and warm_start
        yield SimpleNamespace(execute=execute, set_warm_start_updates=lambda _: None)

    def calculator(**kwargs: typing.Any) -> SimpleNamespace:
        assert kwargs["energy_tolerance"] == 1e-10
        assert kwargs["density_tolerance"] == 1e-8
        assert kwargs["screening_tolerance"] == 1e-12
        return SimpleNamespace(prepare_batch=prepare)

    support = SimpleNamespace(
        raw_output_path=Path,
        environment_metadata=lambda **_: {},
        write_result=lambda *_: pytest.fail("unexpected file publication"),
    )
    monkeypatch.setitem(sys.modules, "_support", support)
    monkeypatch.setitem(sys.modules, "vibeqc", SimpleNamespace(Calculator=calculator))
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None))
            )
        ),
    )
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-cli-audit-not-a-device-run")
    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--batch-sizes", str(batch_size), "--repeats", "1"]
    )
    runpy.run_path(
        str(ROOT / "benchmarks/issue992_pooled_direct_jk.py"), run_name="__main__"
    )


@pytest.mark.parametrize("phase", [0, 1, 2], ids=["cold", "setup", "replay"])
@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "extra",
        "nan",
        "inf",
        "-inf",
        "unconverged",
        "cpu_reference",
        "unknown_backend",
        "missing_backend",
    ],
)
def test_invalid_pooled_results_never_publish_throughput(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    phase: int,
    defect: str,
) -> None:
    with pytest.raises(RuntimeError):
        run_benchmark(monkeypatch, phase=phase, defect=defect)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_valid_pooled_results_preserve_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], batch_size: int
) -> None:
    run_benchmark(monkeypatch, batch_size=batch_size)
    payload = json.loads(capsys.readouterr().out)
    record = payload["results"][str(batch_size)]
    assert record["iterations"] == [[3] * batch_size]
    assert record["warm_systems_per_second"] > 0
    assert record["batch_size"] == batch_size


def test_pooled_energy_spread_gate_remains_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="energy spread"):
        run_benchmark(monkeypatch, phase=2, defect="spread")
