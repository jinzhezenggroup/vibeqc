"""Publication control flow using sentinels, not a CUDA mathematical oracle.

Execute the repository result method without importing native-runtime owners.
"""

from __future__ import annotations

import ast
import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tools/vibeqc_cc/triples_complete_gradient.py"


class PublicationError(RuntimeError):
    pass


class Record(SimpleNamespace):
    """Unused, unchanged diagnostic scalar fields default to zero."""

    def __getattr__(self, name: str) -> int:
        return 0


@pytest.fixture
def publication() -> typing.Any:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BoundCCSDTGradient"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_result"
    )
    environment = {
        "typing": typing,
        "np": np,
        "CCSDGradientResult": lambda *args: args,
        "ImplicitSolveError": PublicationError,
    }
    # Execute only the selected, trusted repository method, not supplied code.
    exec(  # noqa: S102
        compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"),
        environment,
    )
    return environment["_result"], environment


def _owner(
    environment: typing.Any, *, cuda: bool = True, chunk: int | None = 1
) -> typing.Any:
    calls: list[tuple] = []
    arrays = tuple(np.array([float(i)]) for i in range(8))
    bound = Record(
        feeds={}, independent=Record(primal="ccsd-primal"), tensor_backend="cpu"
    )

    def cpu(*args: typing.Any) -> float:
        calls.append(("cpu",))
        if cuda:
            raise AssertionError("CPU triples oracle executed on selected publication")
        return -0.015625

    def validate(nocc: int, nvir: int, *values: typing.Any) -> None:
        assert (nocc, nvir) == (2, 4)
        assert all(x is y for x, y in zip(values, arrays, strict=True))
        calls.append(("validate",))

    def guard(eps_o: typing.Any, eps_v: typing.Any, threshold: float) -> None:
        assert eps_o is arrays[6] and eps_v is arrays[7] and threshold == 1e-10
        calls.append(("denominators",))

    def tiles(nocc: int, nvir: int, *, vir_chunk_size: int) -> typing.Any:
        assert (nocc, nvir) == (2, 4) and vir_chunk_size > 0
        for start in range(0, nvir, vir_chunk_size):
            yield Record(a_start=start, a_end=min(start + vir_chunk_size, nvir))

    def program(nocc: int, nvir: int, *, vir_chunk: tuple[int, int]) -> typing.Any:
        assert (nocc, nvir) == (2, 4)
        return vir_chunk

    def feeds(values: typing.Any, end: int) -> dict[str, int]:
        assert tuple(values) == (
            "ovvv",
            "ovoo",
            "ovov",
            "fov",
            "t1",
            "t2",
            "eps_o",
            "eps_v",
        )
        assert all(x is y for x, y in zip(values.values(), arrays, strict=True))
        return {"end": end}

    def run(program: typing.Any, feeds: typing.Any) -> dict[str, float]:
        start, end = program
        assert feeds == {"end": end}
        calls.append(("execute", start, end))
        return {"triples_energy": -0.00390625 * (end - start)}

    def ccsd(program: typing.Any, feeds: typing.Any) -> dict[str, float]:
        assert program == "ccsd-primal" and feeds is bound.feeds
        return {"correlation_energy": -0.125}

    fixed = Record(
        bound=bound,
        baseline=Record(_execute_tensor=ccsd),
        corrected=Record(),
        vir_chunk_size=chunk,
    )
    state = Record(
        response=fixed,
        baseline=Record(response_backend=Record(identity="jk")),
        z_result=Record(),
    )
    obj = Record(
        response=state,
        reference=Record(nocc=2, nmo=6, reference_energy=-75.0),
        provider=Record(),
        timings={},
        tensor_executor=Record(backend="cuda-fp64-ordinary-stream") if cuda else None,
        _run=run,
        _assert_current=lambda: calls.append(("current",)),
    )
    environment.update(
        triples_energy=cpu,
        _triples_arrays=lambda bound: arrays,
        _validate=validate,
        _check_denominators=guard,
        TriplesTileEnumerator=tiles,
        build_tile_triples_program=program,
        _tile_input_feeds=feeds,
    )
    return obj, calls


def _publish(context: typing.Any, state: typing.Any) -> typing.Any:
    return context[0](state, np.zeros((1, 3)), {}, {}, {})


@pytest.mark.parametrize("chunk", [None, 1, 2, 3])
def test_selected_executor_publishes_all_tiles(
    publication: typing.Any, chunk: int | None
) -> None:
    state, calls = _owner(publication[1], chunk=chunk)
    result = _publish(publication, state)
    step = chunk or 1
    assert [call[1:] for call in calls if call[0] == "execute"] == [
        (start, min(start + step, 4)) for start in range(0, 4, step)
    ]
    assert result[0] == -75.140625 and result[-1]["triples_energy"] == -0.015625
    assert not any(call[0] == "cpu" for call in calls)
    assert calls[:2] == [("validate",), ("denominators",)]


def test_cpu_default_is_unchanged(publication: typing.Any) -> None:
    state, calls = _owner(publication[1], cuda=False)
    assert _publish(publication, state)[0] == -75.140625
    assert sum(call[0] == "cpu" for call in calls) == 1
    assert not any(call[0] == "execute" for call in calls)


def test_selected_executor_budget_failure_has_no_cpu_retry(
    publication: typing.Any,
) -> None:
    state, calls = _owner(publication[1])

    def fail(*args: typing.Any) -> typing.NoReturn:
        raise MemoryError("selected byte budget")

    state._run = fail
    with pytest.raises(MemoryError, match="byte budget"):
        _publish(publication, state)
    assert not any(call[0] == "cpu" for call in calls)


def test_denominator_failure_precedes_execution(publication: typing.Any) -> None:
    state, calls = _owner(publication[1])

    def fail(*args: typing.Any) -> typing.NoReturn:
        raise ValueError("denominator rejected")

    publication[1]["_check_denominators"] = fail
    with pytest.raises(ValueError, match="denominator rejected"):
        _publish(publication, state)
    assert not any(call[0] in ("cpu", "execute") for call in calls)


def test_finite_tiles_cannot_publish_overflowed_sum(publication: typing.Any) -> None:
    state, _ = _owner(publication[1])
    state._run = lambda *args: {"triples_energy": 1e308}
    with pytest.raises(PublicationError, match="nonfinite"):
        _publish(publication, state)


def test_freshness_rechecked_after_energy(publication: typing.Any) -> None:
    state, _ = _owner(publication[1])
    original = state._run
    dirty = False

    def run(*args: typing.Any) -> typing.Any:
        nonlocal dirty
        result = original(*args)
        dirty = True
        return result

    def current() -> None:
        if dirty:
            raise PublicationError("state rebound")

    state._run, state._assert_current = run, current
    with pytest.raises(PublicationError, match="state rebound"):
        _publish(publication, state)


def test_shared_input_guard_is_not_bypassed(publication: typing.Any) -> None:
    state, calls = _owner(publication[1])

    def fail(*args: typing.Any) -> typing.NoReturn:
        raise ValueError("invalid triples input")

    publication[1]["_validate"] = fail
    with pytest.raises(ValueError, match="invalid triples input"):
        _publish(publication, state)
    assert not any(call[0] in ("cpu", "execute", "denominators") for call in calls)


@pytest.mark.parametrize("cuda", (False, True))
@pytest.mark.parametrize(
    "reference,ccsd,triples",
    [
        (-75.0, 1e308, 1e308),
        (1e308, -0.125, 1e308),
        (-75.0, float("nan"), -0.015625),
        (float("inf"), -0.125, -0.015625),
        (-75.0, -0.125, float("inf")),
    ],
)
def test_nonfinite_energy_is_rejected_before_result_construction(
    publication: typing.Any, cuda: bool, reference: float, ccsd: float, triples: float
) -> None:
    state, _ = _owner(publication[1], cuda=cuda)
    state.reference.reference_energy = reference
    state.response.response.baseline._execute_tensor = lambda *args: {
        "correlation_energy": ccsd
    }
    if cuda:
        state._run = lambda *args: {"triples_energy": triples / 4}
    else:
        publication[1]["triples_energy"] = lambda *args: triples

    constructed = []
    publication[1]["CCSDGradientResult"] = lambda *args: constructed.append(args)
    with pytest.raises(PublicationError, match="nonfinite"):
        _publish(publication, state)
    assert constructed == []
