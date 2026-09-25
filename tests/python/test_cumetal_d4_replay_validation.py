"""A positive benchmark duration cannot replace post-replay numerical checks."""

import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "benchmarks/test_cumetal_d4_codspeed.py"
    spec = importlib.util.spec_from_file_location("d4_replay_harness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("reply", ("OK validate d4\n", "", "OK d4 1.0\n", "FAIL\n"))
def test_post_replay_validation_protocol(reply: str) -> None:
    module = _module()
    server = module._D4Server.__new__(module._D4Server)
    sent = io.StringIO()
    server._process = SimpleNamespace(stdin=sent, stdout=io.StringIO(reply))
    if reply == "OK validate d4\n":
        server.validate()
    else:
        with pytest.raises(RuntimeError, match="validation"):
            server.validate()
    assert sent.getvalue() == "validate d4\n"


@pytest.mark.parametrize("valid", (False, True))
def test_final_result_is_checked_outside_the_timed_target(valid: bool) -> None:
    module = _module()
    events = []
    inside = False

    def run_once() -> float:
        assert inside
        events.append("run")
        return 1.0

    def validate() -> None:
        assert not inside
        events.append("validate")
        if not valid:
            raise RuntimeError("D4 replay numerical validation failed")

    def benchmark(target):
        nonlocal inside
        inside = True
        result = target()
        inside = False
        return result

    server = SimpleNamespace(run_once=run_once, validate=validate)
    if valid:
        module.test_cumetal_d4_production_walltime(benchmark, server)
    else:
        with pytest.raises(RuntimeError, match="numerical validation"):
            module.test_cumetal_d4_production_walltime(benchmark, server)
    assert events == ["run", "validate"]


def test_native_post_replay_check_does_not_overwrite_the_measured_result() -> None:
    source = (ROOT / "benchmarks/cumetal_d4_codspeed.cu").read_text()
    check = source.split("bool validate_after_samples()", 1)[1].split(
        "~BenchmarkServer()", 1
    )[0]
    assert "cudaDeviceSynchronize()" in check
    assert "validate_results();" in check
    assert "launch_once" not in check
    assert 'command == "validate d4"' in source
    assert '" status=" + std::to_string' in source
